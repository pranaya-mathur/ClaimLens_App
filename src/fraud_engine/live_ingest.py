"""
Live Claim Ingestion Service - Real-time graph updates
"""
from neo4j import GraphDatabase
from typing import Dict, List
from loguru import logger
from datetime import date


class LiveClaimIngestor:
    """Handles live ingestion of claims into Neo4j graph"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info(f"LiveClaimIngestor connected to Neo4j at {uri}")
    
    def close(self):
        self.driver.close()
        logger.info("LiveClaimIngestor connection closed")
    
    def ingest_claim(self, claim_data: Dict) -> Dict:
        """
        Ingest a single claim with all related entities into the graph.
        Uses MERGE to prevent duplicates and CREATE for new claims.
        
        Args:
            claim_data: Dictionary containing claim, claimant, policy, location, and documents
        
        Returns:
            Dictionary with ingestion statistics
        """
        logger.info(f"Ingesting claim {claim_data['claim_id']}...")
        
        stats = {
            "nodes_created": 0,
            "relationships_created": 0,
            "nodes_updated": 0
        }
        
        with self.driver.session() as session:
            # Single transaction for atomicity
            result = session.execute_write(
                self._create_claim_graph,
                claim_data
            )
            stats.update(result)
        
        logger.success(f"✓ Claim {claim_data['claim_id']} ingested: {stats['nodes_created']} nodes, {stats['relationships_created']} relationships")
        return stats
    
    @staticmethod
    def _create_claim_graph(tx, claim_data: Dict) -> Dict:
        """
        Create/update all nodes and relationships for a claim in one transaction.
        """
        stats = {
            "nodes_created": 0,
            "relationships_created": 0,
            "nodes_updated": 0
        }
        
        # Calculate days since policy start
        policy_start = claim_data['policy']['start_date']
        incident_date = claim_data['incident_date']
        if isinstance(policy_start, str):
            policy_start = date.fromisoformat(policy_start)
        if isinstance(incident_date, str):
            incident_date = date.fromisoformat(incident_date)
        days_since_policy = (incident_date - policy_start).days
        
        # Calculate filing delay
        claim_date = claim_data['claim_date']
        if isinstance(claim_date, str):
            claim_date = date.fromisoformat(claim_date)
        filing_delay = (claim_date - incident_date).days
        
        # Main query - creates all nodes and relationships
        query = """
        // 1. Create or merge Claimant
        MERGE (claimant:Claimant {claimant_id: $claimant_id})
        ON CREATE SET
            claimant.name = $claimant_name,
            claimant.phone = $claimant_phone,
            claimant.email = $claimant_email,
            claimant.city = $claimant_city,
            claimant.total_claims = 1,
            claimant.fraud_count = $fraud_history,
            claimant.is_high_risk = CASE WHEN $fraud_history > 0 THEN true ELSE false END,
            claimant.created_at = datetime()
        ON MATCH SET
            claimant.total_claims = claimant.total_claims + 1,
            claimant.last_claim_date = datetime()
        
        // 2. Merge Policy
        MERGE (policy:Policy {policy_number: $policy_number})
        ON CREATE SET
            policy.product_type = $policy_product_type,
            policy.sum_insured = $sum_insured,
            policy.start_date = date($policy_start_date),
            policy.claim_count = 1,
            policy.total_claimed = $claim_amount,
            policy.created_at = datetime()
        ON MATCH SET
            policy.claim_count = policy.claim_count + 1,
            policy.total_claimed = policy.total_claimed + $claim_amount
        
        // 3. Merge City/Location
        MERGE (city:City {city_name: $city_name})
        ON CREATE SET
            city.state = $state,
            city.claim_count = 1
        ON MATCH SET
            city.claim_count = city.claim_count + 1
        
        // 4. Create new Claim (always new)
        CREATE (claim:Claim {
            claim_id: $claim_id,
            product: $product_type,
            amount: $claim_amount,
            subtype: $subtype,
            incident_date: date($incident_date),
            claim_date: date($claim_date),
            filing_delay_days: $filing_delay,
            days_since_policy_start: $days_since_policy,
            is_delayed_filing: CASE WHEN $filing_delay > 7 THEN true ELSE false END,
            status: 'pending_review',
            created_at: datetime()
        })
        
        // 5. Create relationships
        CREATE (claim)-[:FILED_BY]->(claimant)
        CREATE (claim)-[:ON_POLICY]->(policy)
        CREATE (claim)-[:IN_LOCATION]->(city)
        
        // 6. Link claimant to policy if not already linked
        MERGE (claimant)-[:HAS_POLICY]->(policy)
        
        // Return counts
        RETURN 
            1 as claim_created,
            CASE WHEN claimant.total_claims = 1 THEN 1 ELSE 0 END as claimant_created,
            CASE WHEN policy.claim_count = 1 THEN 1 ELSE 0 END as policy_created,
            CASE WHEN city.claim_count = 1 THEN 1 ELSE 0 END as city_created
        """
        
        params = {
            # Claim params
            "claim_id": claim_data['claim_id'],
            "claim_amount": float(claim_data['claim_amount']),
            "claim_date": str(claim_data['claim_date']),
            "incident_date": str(claim_data['incident_date']),
            "product_type": claim_data['product_type'],
            "subtype": claim_data.get('subtype', ''),
            "filing_delay": filing_delay,
            "days_since_policy": days_since_policy,
            
            # Claimant params
            "claimant_id": claim_data['claimant']['claimant_id'],
            "claimant_name": claim_data['claimant']['name'],
            "claimant_phone": claim_data['claimant']['phone'],
            "claimant_email": claim_data['claimant'].get('email', ''),
            "claimant_city": claim_data['claimant']['city'],
            "fraud_history": claim_data['claimant'].get('fraud_history', 0),
            
            # Policy params
            "policy_number": claim_data['policy']['policy_number'],
            "policy_product_type": claim_data['policy']['product_type'],
            "sum_insured": float(claim_data['policy']['sum_insured']),
            "policy_start_date": str(claim_data['policy']['start_date']),
            
            # Location params
            "city_name": claim_data['location']['city_name'],
            "state": claim_data['location']['state']
        }
        
        result = tx.run(query, **params)
        record = result.single()
        
        # Calculate stats
        stats['nodes_created'] = (
            record['claim_created'] +
            record['claimant_created'] +
            record['policy_created'] +
            record['city_created']
        )
        stats['relationships_created'] = 4  # FILED_BY, ON_POLICY, IN_LOCATION, HAS_POLICY
        
        # Process documents if provided
        if claim_data.get('documents'):
            doc_stats = LiveClaimIngestor._create_documents(
                tx,
                claim_data['claim_id'],
                claim_data['documents'],
                filing_delay
            )
            stats['nodes_created'] += doc_stats['docs_created']
            stats['relationships_created'] += doc_stats['docs_created']  # Each doc has HAS_DOCUMENT relationship
        
        return stats
    
    @staticmethod
    def _create_documents(tx, claim_id: str, documents: List[Dict], filing_delay: int) -> Dict:
        """
        Create/merge document nodes and link to claim.
        Documents are identified by doc_hash to detect reuse across claims.
        """
        if not documents:
            return {"docs_created": 0}
        
        query = """
        UNWIND $documents AS doc
        
        // Merge document by hash (to detect reuse)
        MERGE (d:Document {doc_hash: doc.doc_hash})
        ON CREATE SET
            d.doc_id = doc.doc_id,
            d.doc_type = doc.doc_type,
            d.first_used_claim = $claim_id,
            d.usage_count = 1,
            d.is_suspicious = false,
            d.created_at = datetime()
        ON MATCH SET
            d.usage_count = d.usage_count + 1,
            d.is_suspicious = true,  // Mark as suspicious if reused
            d.last_used_claim = $claim_id
        
        // Link document to claim
        WITH d
        MATCH (c:Claim {claim_id: $claim_id})
        CREATE (c)-[:HAS_DOCUMENT {
            is_delayed: CASE WHEN $filing_delay > 7 THEN true ELSE false END,
            attached_at: datetime()
        }]->(d)
        
        RETURN CASE WHEN d.usage_count = 1 THEN 1 ELSE 0 END as doc_created
        """
        
        result = tx.run(
            query,
            claim_id=claim_id,
            documents=documents,
            filing_delay=filing_delay
        )
        
        docs_created = sum(record['doc_created'] for record in result)
        return {"docs_created": docs_created}
    
    def check_claim_exists(self, claim_id: str) -> bool:
        """
        Check if a claim already exists in the graph.
        """
        with self.driver.session() as session:
            result = session.run(
                "MATCH (c:Claim {claim_id: $claim_id}) RETURN count(c) as count",
                claim_id=claim_id
            )
            record = result.single()
            return record['count'] > 0
