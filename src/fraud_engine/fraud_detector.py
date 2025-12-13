"""
Fraud Detector - Query Neo4j for fraud patterns
"""
from neo4j import GraphDatabase
from typing import Dict, List, Union
from loguru import logger


class FraudDetector:
    """Detect fraud patterns using graph queries"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def get_graph_risk_score(self, claim_id: Union[int, str]) -> Dict:
        """
        Calculate graph-based risk score for a claim
        
        Checks:
        - Claimant's fraud history
        - Neighbor claims' fraud rate
        - Document sharing patterns
        - Policy abuse signals
        
        Args:
            claim_id: Claim identifier (can be int or str)
        """
        query = """
        MATCH (c:Claim {claim_id: $claim_id})-[:FILED_BY]->(p:Claimant)
        
        // Get claimant fraud history
        OPTIONAL MATCH (p)<-[:FILED_BY]-(other:Claim)
        WHERE other.fraud_label = 1 AND other.claim_id <> $claim_id
        WITH c, p, COUNT(other) AS neighbor_fraud_count
        
        // Check for document sharing
        OPTIONAL MATCH (c)-[:HAS_DOCUMENT]->(d:Document)<-[:HAS_DOCUMENT]-(dup:Claim)
        WHERE c.claim_id <> dup.claim_id
        WITH c, p, neighbor_fraud_count, COUNT(DISTINCT dup) AS doc_sharing_count
        
        // Check policy age
        OPTIONAL MATCH (c)-[r:ON_POLICY]->(pol:Policy)
        
        RETURN 
            c.claim_id AS claim_id,
            c.fraud_score AS base_score,
            c.amount AS claim_amount,
            c.days_since_policy_start AS policy_age,
            c.filing_delay_days AS filing_delay,
            p.fraud_rate AS claimant_fraud_rate,
            p.total_claims AS claimant_total_claims,
            p.fraud_count AS claimant_fraud_count,
            neighbor_fraud_count,
            doc_sharing_count,
            pol.claim_count AS policy_claim_count
        """
        
        with self.driver.session() as session:
            result = session.run(query, claim_id=claim_id).single()
        
        if not result:
            return {"error": f"Claim {claim_id} not found in graph"}
        
        # Calculate base score from available data if not present
        base_score = result['base_score'] if result['base_score'] is not None else 0.0
        
        # If no fraud_score in claim, calculate basic score from attributes
        if base_score == 0.0:
            filing_delay = result['filing_delay'] or 0
            policy_age = result['policy_age'] or 0
            claim_amount = result['claim_amount'] or 0
            
            # Simple heuristic scoring for live ingested claims
            delay_risk = min(filing_delay / 30.0, 0.3)  # Max 0.3 for delays
            new_policy_risk = 0.2 if policy_age <= 30 else 0.0
            high_amount_risk = 0.2 if claim_amount > 400000 else 0.0
            
            base_score = delay_risk + new_policy_risk + high_amount_risk
        
        claimant_fraud_rate = result['claimant_fraud_rate'] or 0.0
        claimant_fraud_count = result['claimant_fraud_count'] or 0
        neighbor_frauds = result['neighbor_fraud_count'] or 0
        doc_sharing = result['doc_sharing_count'] or 0
        policy_age = result['policy_age'] or 0
        
        # Risk components
        graph_risk = min(neighbor_frauds * 0.15, 0.5)  # Max 0.5 from neighbors
        doc_risk = min(doc_sharing * 0.2, 0.3)  # Max 0.3 from doc sharing
        claimant_risk = min(claimant_fraud_count * 0.1, 0.3)  # Max 0.3 from history
        new_policy_risk = 0.2 if policy_age <= 30 and base_score > 0.5 else 0.0
        
        # Combined score
        combined_score = (
            base_score * 0.3 +
            claimant_fraud_rate * 0.15 +
            claimant_risk * 0.15 +
            graph_risk +
            doc_risk +
            new_policy_risk
        )
        
        return {
            "claim_id": str(result['claim_id']),
            "base_fraud_score": round(base_score, 3),
            "claimant_fraud_rate": round(claimant_fraud_rate, 3),
            "claimant_fraud_count": claimant_fraud_count,
            "neighbor_fraud_count": neighbor_frauds,
            "doc_sharing_count": doc_sharing,
            "graph_risk_component": round(graph_risk, 3),
            "final_risk_score": round(min(combined_score, 1.0), 3),
            "risk_level": self._get_risk_level(combined_score)
        }
    
    @staticmethod
    def _get_risk_level(score: float) -> str:
        if score >= 0.8:
            return "CRITICAL"
        elif score >= 0.6:
            return "HIGH"
        elif score >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def find_fraud_rings(self, min_shared_docs: int = 2) -> List[Dict]:
        """Find fraud rings based on shared documents"""
        query = """
        MATCH (c1:Claim)-[:HAS_DOCUMENT]->(d:Document)<-[:HAS_DOCUMENT]-(c2:Claim)
        WHERE c1.claim_id < c2.claim_id
        MATCH (c1)-[:FILED_BY]->(p1:Claimant)
        MATCH (c2)-[:FILED_BY]->(p2:Claimant)
        WHERE p1.claimant_id <> p2.claimant_id
        WITH p1, p2, 
             COUNT(DISTINCT d) AS shared_docs,
             AVG(COALESCE(c1.fraud_score, 0.0) + COALESCE(c2.fraud_score, 0.0)) / 2 AS avg_fraud_score,
             COLLECT(DISTINCT c1.claim_id) + COLLECT(DISTINCT c2.claim_id) AS claim_ids
        WHERE shared_docs >= $min_shared
        RETURN 
            p1.claimant_id AS claimant_1,
            p2.claimant_id AS claimant_2,
            shared_docs,
            avg_fraud_score,
            claim_ids
        ORDER BY shared_docs DESC, avg_fraud_score DESC
        LIMIT 50
        """
        
        with self.driver.session() as session:
            results = session.run(query, min_shared=min_shared_docs).data()
        
        return results
    
    def find_serial_fraudsters(self, min_fraud_claims: int = 3) -> List[Dict]:
        """Find claimants with multiple high-fraud claims or high fraud_count"""
        query = """
        MATCH (p:Claimant)<-[:FILED_BY]-(c:Claim)
        WHERE COALESCE(c.fraud_score, 0.0) > 0.7 OR p.fraud_count > 0
        WITH p, 
             COUNT(c) AS high_fraud_claims,
             AVG(COALESCE(c.fraud_score, 0.0)) AS avg_score,
             SUM(c.amount) AS total_claimed,
             COLLECT(c.claim_id) AS claim_ids,
             p.fraud_count AS known_frauds
        WHERE high_fraud_claims >= $min_claims OR known_frauds >= $min_claims
        RETURN 
            p.claimant_id,
            high_fraud_claims,
            known_frauds,
            avg_score,
            total_claimed,
            claim_ids
        ORDER BY known_frauds DESC, high_fraud_claims DESC, avg_score DESC
        LIMIT 30
        """
        
        with self.driver.session() as session:
            results = session.run(query, min_claims=min_fraud_claims).data()
        
        return results
    
    def detect_policy_abuse(self) -> List[Dict]:
        """Find new policy + immediate high claims"""
        query = """
        MATCH (c:Claim)-[r:ON_POLICY]->(pol:Policy)
        WHERE c.days_since_policy_start <= 30
          AND c.amount > 400000
          AND (COALESCE(c.fraud_score, 0.0) > 0.5 OR c.filing_delay_days > 7)
        MATCH (c)-[:FILED_BY]->(p:Claimant)
        RETURN 
            c.claim_id,
            p.claimant_id,
            c.amount,
            c.days_since_policy_start AS days_old,
            c.filing_delay_days AS filing_delay,
            COALESCE(c.fraud_score, 0.0) AS fraud_score
        ORDER BY c.amount DESC
        LIMIT 20
        """
        
        with self.driver.session() as session:
            results = session.run(query).data()
        
        return results
