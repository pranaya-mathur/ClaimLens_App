"""Claim Storage Layer for Neo4j Database

Stores analyzed claims in Neo4j for:
- Future graph-based fraud detection
- Fraud ring identification
- Serial fraudster detection
- Network analysis
"""
from neo4j import GraphDatabase
from typing import Dict, Optional, List
from datetime import datetime
from loguru import logger
import uuid


class ClaimStorage:
    """Store and manage claims in Neo4j graph database."""
    
    def __init__(self, uri: str, user: str, password: str):
        """Initialize Neo4j connection.
        
        Args:
            uri: Neo4j connection URI (bolt://localhost:7687)
            user: Database username
            password: Database password
        """
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            logger.info(f"✅ Connected to Neo4j at {uri}")
        except Exception as e:
            logger.error(f"❌ Neo4j connection failed: {e}")
            raise
    
    def close(self):
        """Close database connection."""
        if self.driver:
            self.driver.close()
    
    def store_claim(
        self,
        claim_data: Dict,
        ml_result: Dict,
        cv_result: Optional[Dict] = None,
        graph_result: Optional[Dict] = None,
        llm_result: Optional[Dict] = None
    ) -> Dict:
        """Store new claim with analysis results in Neo4j.
        
        Args:
            claim_data: Raw claim information
            ml_result: ML fraud detection results
            cv_result: Computer vision document verification
            graph_result: Graph-based fraud detection
            llm_result: LLM semantic aggregation
            
        Returns:
            Dict with storage status and claim node ID
        """
        try:
            with self.driver.session() as session:
                result = session.execute_write(
                    self._create_claim_transaction,
                    claim_data,
                    ml_result,
                    cv_result,
                    graph_result,
                    llm_result
                )
                logger.success(f"✅ Stored claim {claim_data['claim_id']} in Neo4j")
                return result
        except Exception as e:
            logger.error(f"❌ Failed to store claim {claim_data.get('claim_id')}: {e}")
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def _create_claim_transaction(tx, claim_data, ml_result, cv_result, graph_result, llm_result):
        """Transaction to create claim node and relationships."""
        
        # Generate unique internal ID if not exists
        internal_id = str(uuid.uuid4())
        
        # Create Claim node
        claim_query = """
        MERGE (c:Claim {claim_id: $claim_id})
        ON CREATE SET
            c.internal_id = $internal_id,
            c.product = $product,
            c.subtype = $subtype,
            c.claim_amount = $claim_amount,
            c.days_since_policy_start = $days_since_policy_start,
            c.narrative = $narrative,
            c.incident_date = $incident_date,
            c.created_at = datetime(),
            c.ml_fraud_score = $ml_fraud_score,
            c.ml_risk_level = $ml_risk_level,
            c.final_verdict = $final_verdict,
            c.final_confidence = $final_confidence,
            c.cv_verified = $cv_verified,
            c.llm_used = $llm_used
        ON MATCH SET
            c.updated_at = datetime(),
            c.ml_fraud_score = $ml_fraud_score,
            c.ml_risk_level = $ml_risk_level,
            c.final_verdict = $final_verdict,
            c.final_confidence = $final_confidence
        RETURN c.internal_id as id, c.claim_id as claim_id
        """
        
        claim_params = {
            "claim_id": claim_data["claim_id"],
            "internal_id": internal_id,
            "product": claim_data.get("product", "unknown"),
            "subtype": claim_data.get("subtype", "unknown"),
            "claim_amount": float(claim_data.get("claim_amount", 0)),
            "days_since_policy_start": int(claim_data.get("days_since_policy_start", 0)),
            "narrative": claim_data.get("narrative", ""),
            "incident_date": claim_data.get("incident_date", str(datetime.now().date())),
            "ml_fraud_score": float(ml_result.get("fraud_probability", 0)),
            "ml_risk_level": ml_result.get("risk_level", "UNKNOWN"),
            "final_verdict": llm_result.get("verdict", "REVIEW") if llm_result else "REVIEW",
            "final_confidence": float(llm_result.get("confidence", 0.5)) if llm_result else 0.5,
            "cv_verified": cv_result is not None and cv_result.get("risk_score", 1) < 0.4,
            "llm_used": llm_result is not None and llm_result.get("llm_used", False)
        }
        
        claim_result = tx.run(claim_query, claim_params).single()
        
        # Create/Link Claimant
        claimant_query = """
        MERGE (cl:Claimant {claimant_id: $claimant_id})
        ON CREATE SET cl.created_at = datetime()
        WITH cl
        MATCH (c:Claim {claim_id: $claim_id})
        MERGE (cl)-[:FILED]->(c)
        """
        
        tx.run(claimant_query, {
            "claimant_id": claim_data.get("claimant_id", "UNKNOWN"),
            "claim_id": claim_data["claim_id"]
        })
        
        # Create/Link Policy
        policy_query = """
        MERGE (p:Policy {policy_id: $policy_id})
        ON CREATE SET p.created_at = datetime()
        WITH p
        MATCH (c:Claim {claim_id: $claim_id})
        MERGE (c)-[:UNDER_POLICY]->(p)
        """
        
        tx.run(policy_query, {
            "policy_id": claim_data.get("policy_id", "UNKNOWN"),
            "claim_id": claim_data["claim_id"]
        })
        
        # Link Documents if provided
        documents = claim_data.get("documents_submitted", "")
        if documents:
            doc_list = [d.strip() for d in documents.split(",")]
            for doc in doc_list:
                doc_query = """
                MERGE (d:Document {document_type: $doc_type})
                WITH d
                MATCH (c:Claim {claim_id: $claim_id})
                MERGE (c)-[:SUBMITTED_DOCUMENT]->(d)
                """
                tx.run(doc_query, {
                    "doc_type": doc,
                    "claim_id": claim_data["claim_id"]
                })
        
        return {
            "success": True,
            "claim_id": claim_result["claim_id"],
            "internal_id": claim_result["id"],
            "stored_at": datetime.now().isoformat()
        }
    
    def get_claimant_history(self, claimant_id: str) -> Dict:
        """Get claim history for a claimant.
        
        Args:
            claimant_id: Claimant identifier
            
        Returns:
            Dict with claim count, fraud rate, total amount
        """
        query = """
        MATCH (cl:Claimant {claimant_id: $claimant_id})-[:FILED]->(c:Claim)
        RETURN 
            count(c) as total_claims,
            avg(c.ml_fraud_score) as avg_fraud_score,
            sum(c.claim_amount) as total_amount,
            collect(c.claim_id) as claim_ids
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query, {"claimant_id": claimant_id}).single()
                
                if result:
                    return {
                        "claimant_id": claimant_id,
                        "total_claims": result["total_claims"],
                        "avg_fraud_score": float(result["avg_fraud_score"] or 0),
                        "total_amount": float(result["total_amount"] or 0),
                        "claim_ids": result["claim_ids"]
                    }
                else:
                    return {
                        "claimant_id": claimant_id,
                        "total_claims": 0,
                        "avg_fraud_score": 0,
                        "total_amount": 0,
                        "claim_ids": []
                    }
        except Exception as e:
            logger.error(f"Failed to get claimant history: {e}")
            return {"error": str(e)}
    
    def find_fraud_connections(self, claim_id: str, max_depth: int = 2) -> Dict:
        """Find fraud connections for a claim.
        
        Args:
            claim_id: Claim identifier
            max_depth: Maximum relationship depth to search
            
        Returns:
            Dict with connected claims and fraud indicators
        """
        query = """
        MATCH (c:Claim {claim_id: $claim_id})
        OPTIONAL MATCH (c)-[:FILED|UNDER_POLICY|SUBMITTED_DOCUMENT*1..2]-(connected:Claim)
        WHERE connected.ml_fraud_score > 0.5
        RETURN 
            count(DISTINCT connected) as fraud_connections,
            collect(DISTINCT connected.claim_id) as connected_fraud_claims
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query, {"claim_id": claim_id}).single()
                
                if result:
                    return {
                        "claim_id": claim_id,
                        "fraud_connections": result["fraud_connections"],
                        "connected_fraud_claims": result["connected_fraud_claims"]
                    }
                else:
                    return {
                        "claim_id": claim_id,
                        "fraud_connections": 0,
                        "connected_fraud_claims": []
                    }
        except Exception as e:
            logger.error(f"Failed to find fraud connections: {e}")
            return {"error": str(e)}
