"""
Analytics API Routes
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict

from src.fraud_engine.fraud_detector import FraudDetector
from config.settings import get_settings


router = APIRouter()


def get_fraud_detector():
    """Dependency to get fraud detector instance"""
    settings = get_settings()
    return FraudDetector(
        uri=settings.NEO4J_URI,
        user=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD
    )


@router.get("/overview")
def get_overview_stats(
    detector: FraudDetector = Depends(get_fraud_detector)
):
    """Get overall fraud statistics"""
    query = """
    MATCH (c:Claim)
    WITH COUNT(c) AS total_claims,
         SUM(CASE WHEN c.fraud_label = 1 THEN 1 ELSE 0 END) AS fraud_claims,
         AVG(c.fraud_score) AS avg_fraud_score,
         SUM(c.amount) AS total_amount
    RETURN total_claims, fraud_claims, avg_fraud_score, total_amount
    """
    
    try:
        with detector.driver.session() as session:
            result = session.run(query).single()
        
        total = result['total_claims']
        fraud = result['fraud_claims']
        
        return {
            "total_claims": total,
            "fraud_claims": fraud,
            "fraud_rate": round(fraud / total * 100, 2) if total > 0 else 0,
            "avg_fraud_score": round(result['avg_fraud_score'], 3),
            "total_amount": round(result['total_amount'], 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        detector.close()


@router.get("/by-product")
def get_fraud_by_product(
    detector: FraudDetector = Depends(get_fraud_detector)
):
    """Get fraud statistics by product type"""
    query = """
    MATCH (c:Claim)
    WITH c.product AS product,
         COUNT(c) AS total,
         SUM(CASE WHEN c.fraud_label = 1 THEN 1 ELSE 0 END) AS frauds
    WHERE product IS NOT NULL
    RETURN product, total, frauds,
           CASE WHEN total > 0 THEN toFloat(frauds) / total * 100 ELSE 0 END AS fraud_rate
    ORDER BY fraud_rate DESC
    """
    
    try:
        with detector.driver.session() as session:
            results = session.run(query).data()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        detector.close()


@router.get("/by-city")
def get_fraud_by_city(
    detector: FraudDetector = Depends(get_fraud_detector)
):
    """Get fraud statistics by city (top 10)"""
    query = """
    MATCH (c:Claim)-[:IN_LOCATION]->(city:City)
    WITH city.city_name AS city,
         COUNT(c) AS total,
         SUM(CASE WHEN c.fraud_label = 1 THEN 1 ELSE 0 END) AS frauds
    WHERE total >= 50
    RETURN city, total, frauds,
           CASE WHEN total > 0 THEN toFloat(frauds) / total * 100 ELSE 0 END AS fraud_rate
    ORDER BY fraud_rate DESC
    LIMIT 10
    """
    
    try:
        with detector.driver.session() as session:
            results = session.run(query).data()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        detector.close()


@router.get("/risk-distribution")
def get_risk_distribution(
    detector: FraudDetector = Depends(get_fraud_detector)
):
    """Get distribution of claims by risk level"""
    query = """
    MATCH (c:Claim)
    WITH CASE 
        WHEN c.fraud_score >= 0.8 THEN 'CRITICAL'
        WHEN c.fraud_score >= 0.6 THEN 'HIGH'
        WHEN c.fraud_score >= 0.4 THEN 'MEDIUM'
        ELSE 'LOW'
    END AS risk_level,
    COUNT(c) AS count
    RETURN risk_level, count
    ORDER BY 
        CASE risk_level
            WHEN 'CRITICAL' THEN 1
            WHEN 'HIGH' THEN 2
            WHEN 'MEDIUM' THEN 3
            ELSE 4
        END
    """
    
    try:
        with detector.driver.session() as session:
            results = session.run(query).data()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        detector.close()