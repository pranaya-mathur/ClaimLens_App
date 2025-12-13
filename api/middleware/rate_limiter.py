"""
Rate Limiting Middleware
Protects API endpoints from abuse and DoS attacks
"""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, Optional
import os
from loguru import logger


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiter.
    
    Configuration via environment variables:
    - RATE_LIMIT_REQUESTS: Max requests per window (default: 100)
    - RATE_LIMIT_WINDOW_SECONDS: Time window in seconds (default: 60)
    - RATE_LIMIT_ENABLED: Enable/disable rate limiting (default: true)
    
    For production, consider using Redis-based rate limiting (e.g., slowapi).
    """
    
    def __init__(self, app):
        super().__init__(app)
        
        # Configuration
        self.enabled = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
        self.max_requests = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
        self.window_seconds = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
        
        # In-memory storage (IP -> list of timestamps)
        self.request_history: Dict[str, list] = defaultdict(list)
        
        # Cleanup interval
        self.last_cleanup = datetime.now()
        self.cleanup_interval = timedelta(minutes=5)
        
        if self.enabled:
            logger.info(
                f"Rate limiting enabled: {self.max_requests} requests "
                f"per {self.window_seconds}s window"
            )
        else:
            logger.warning("Rate limiting DISABLED")
    
    async def dispatch(self, request: Request, call_next):
        """
        Process each request through rate limiter.
        
        Args:
            request: FastAPI request
            call_next: Next middleware/handler
            
        Returns:
            Response or 429 error if rate limit exceeded
        """
        # Skip if disabled
        if not self.enabled:
            return await call_next(request)
        
        # Skip health check endpoints
        if request.url.path in ["/health", "/api/health", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        # Get client IP
        client_ip = self._get_client_ip(request)
        
        # Check rate limit
        if self._is_rate_limited(client_ip):
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.max_requests} requests per {self.window_seconds} seconds. "
                               f"Please slow down and try again later.",
                    "retry_after": self.window_seconds
                },
                headers={
                    "Retry-After": str(self.window_seconds),
                    "X-RateLimit-Limit": str(self.max_requests),
                    "X-RateLimit-Window": str(self.window_seconds)
                }
            )
        
        # Record this request
        self.request_history[client_ip].append(datetime.now())
        
        # Periodic cleanup
        self._cleanup_old_entries()
        
        # Add rate limit headers to response
        response = await call_next(request)
        
        # Calculate remaining requests
        remaining = self._get_remaining_requests(client_ip)
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(self.window_seconds)
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """
        Extract client IP from request.
        
        Handles X-Forwarded-For and X-Real-IP headers for proxies.
        
        Args:
            request: FastAPI request
            
        Returns:
            Client IP address
        """
        # Check proxy headers first
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # X-Forwarded-For can be comma-separated, take first
            return forwarded.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
        
        # Fallback to direct connection IP
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _is_rate_limited(self, client_ip: str) -> bool:
        """
        Check if client has exceeded rate limit.
        
        Args:
            client_ip: Client IP address
            
        Returns:
            True if rate limit exceeded
        """
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.window_seconds)
        
        # Get requests within current window
        recent_requests = [
            ts for ts in self.request_history[client_ip]
            if ts > cutoff
        ]
        
        # Update history to only recent requests
        self.request_history[client_ip] = recent_requests
        
        # Check if limit exceeded
        return len(recent_requests) >= self.max_requests
    
    def _get_remaining_requests(self, client_ip: str) -> int:
        """
        Get remaining requests for client.
        
        Args:
            client_ip: Client IP address
            
        Returns:
            Number of remaining requests in current window
        """
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.window_seconds)
        
        recent_count = sum(
            1 for ts in self.request_history[client_ip]
            if ts > cutoff
        )
        
        return max(0, self.max_requests - recent_count)
    
    def _cleanup_old_entries(self):
        """
        Periodic cleanup of old entries to prevent memory bloat.
        
        Runs every 5 minutes by default.
        """
        now = datetime.now()
        
        if now - self.last_cleanup < self.cleanup_interval:
            return
        
        logger.debug("Running rate limiter cleanup...")
        
        cutoff = now - timedelta(seconds=self.window_seconds * 2)  # Keep 2x window
        
        # Remove old entries
        ips_to_remove = []
        for ip, timestamps in self.request_history.items():
            # Filter out old timestamps
            recent = [ts for ts in timestamps if ts > cutoff]
            
            if recent:
                self.request_history[ip] = recent
            else:
                # Mark for removal if no recent requests
                ips_to_remove.append(ip)
        
        # Remove IPs with no recent activity
        for ip in ips_to_remove:
            del self.request_history[ip]
        
        if ips_to_remove:
            logger.debug(f"Cleaned up {len(ips_to_remove)} inactive IPs from rate limiter")
        
        self.last_cleanup = now


# Optional: Create a decorator for specific endpoint rate limits
class EndpointRateLimiter:
    """
    Decorator for endpoint-specific rate limits.
    
    Usage:
        @router.post("/expensive-operation")
        @rate_limit(max_requests=10, window_seconds=60)
        async def expensive_operation():
            ...
    """
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_history: Dict[str, list] = defaultdict(list)
    
    def __call__(self, func):
        async def wrapper(*args, **kwargs):
            # Extract request from kwargs
            request: Optional[Request] = kwargs.get("request")
            
            if not request:
                # Try to find Request in args
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break
            
            if request:
                client_ip = self._get_client_ip(request)
                
                if self._is_rate_limited(client_ip):
                    raise HTTPException(
                        status_code=429,
                        detail=f"Rate limit exceeded for this endpoint. "
                               f"Max {self.max_requests} requests per {self.window_seconds}s."
                    )
                
                self.request_history[client_ip].append(datetime.now())
            
            return await func(*args, **kwargs)
        
        return wrapper
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP from request."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _is_rate_limited(self, client_ip: str) -> bool:
        """Check if rate limited."""
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.window_seconds)
        
        recent_requests = [
            ts for ts in self.request_history[client_ip]
            if ts > cutoff
        ]
        
        self.request_history[client_ip] = recent_requests
        
        return len(recent_requests) >= self.max_requests


# Convenience function to create decorator
def rate_limit(max_requests: int = 10, window_seconds: int = 60):
    """
    Create a rate limit decorator for endpoint.
    
    Args:
        max_requests: Maximum requests per window
        window_seconds: Time window in seconds
        
    Returns:
        Decorator function
    """
    return EndpointRateLimiter(max_requests, window_seconds)
