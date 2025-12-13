"""
Fraud Graph Loader - Loads claims data into Neo4j
"""
from neo4j import GraphDatabase
import pandas as pd
from typing import List, Dict
from loguru import logger


class FraudGraphLoader:
    """Loads fraud graph into Neo4j"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info(f"Connected to Neo4j at {uri}")
    
    def close(self):
        self.driver.close()
        logger.info("Neo4j connection closed")
    
    def load_claims(self, claims_df: pd.DataFrame, batch_size: int = 1000):
        """Load claims as nodes"""
        logger.info(f"Loading {len(claims_df)} claims...")
        
        total_loaded = 0
        for i in range(0, len(claims_df), batch_size):
            batch = claims_df.iloc[i:i+batch_size]
            claims_data = batch.to_dict('records')
            
            with self.driver.session() as session:
                session.execute_write(self._create_claim_batch, claims_data)
            
            total_loaded += len(claims_data)
            if total_loaded % 5000 == 0:
                logger.info(f"  Loaded {total_loaded:,} claims")
        
        logger.success(f"✓ Loaded {total_loaded:,} claims")
    
    @staticmethod
    def _create_claim_batch(tx, claims: List[Dict]):
        query = """
        UNWIND $claims AS claim
        CREATE (c:Claim {
            claim_id: claim.claim_id,
            product: claim.product,
            amount: toFloat(claim.claim_amount),
            fraud_score: toFloat(claim.fraud_score),
            fraud_label: toInteger(claim.fraud_label),
            red_flags: claim.red_flags,
            subtype: claim.subtype,
            city: claim.city,
            incident_date: date(claim.incident_date),
            days_since_policy_start: toInteger(claim.days_since_policy_start)
        })
        """
        tx.run(query, claims=claims)
    
    def load_claimants(self, claimant_profiles: pd.DataFrame):
        """Load claimant profiles"""
        logger.info(f"Loading {len(claimant_profiles)} claimants...")
        
        claimants = claimant_profiles.to_dict('records')
        with self.driver.session() as session:
            session.execute_write(self._create_claimant_batch, claimants)
        
        logger.success(f"✓ Loaded {len(claimants):,} claimants")
    
    @staticmethod
    def _create_claimant_batch(tx, claimants: List[Dict]):
        query = """
        UNWIND $claimants AS p
        CREATE (person:Claimant {
            claimant_id: p.claimant_id,
            total_claims: toInteger(p.total_claims),
            fraud_rate: toFloat(p.fraud_rate),
            avg_fraud_score: toFloat(p.avg_fraud_score),
            is_high_risk: p.is_high_risk,
            is_repeat: p.is_repeat_claimant
        })
        """
        tx.run(query, claimants=claimants)
    
    def load_policies(self, policy_profiles: pd.DataFrame):
        """Load policy nodes"""
        logger.info(f"Loading {len(policy_profiles)} policies...")
        
        policies = policy_profiles.to_dict('records')
        with self.driver.session() as session:
            session.execute_write(self._create_policy_batch, policies)
        
        logger.success(f"✓ Loaded {len(policies):,} policies")
    
    @staticmethod
    def _create_policy_batch(tx, policies: List[Dict]):
        query = """
        UNWIND $policies AS pol
        CREATE (p:Policy {
            policy_id: pol.policy_id,
            claim_count: toInteger(pol.claim_count),
            unique_claimants: toInteger(pol.unique_claimants),
            total_amount: toFloat(pol.total_amount)
        })
        """
        tx.run(query, policies=policies)
    
    def load_documents(self, docs_df: pd.DataFrame):
        """Load document nodes"""
        logger.info(f"Loading {len(docs_df)} documents...")
        
        docs = docs_df.to_dict('records')
        with self.driver.session() as session:
            session.execute_write(self._create_document_batch, docs)
        
        logger.success(f"✓ Loaded {len(docs):,} documents")
    
    @staticmethod
    def _create_document_batch(tx, docs: List[Dict]):
        query = """
        UNWIND $docs AS doc
        CREATE (d:Document {
            file_name: doc.file_name,
            doc_type: doc.doc_type,
            filing_delay: toInteger(doc.filing_delay_days),
            is_delayed: doc.is_delayed_filing,
            incident_date: date(doc.incident_date),
            filed_date: date(doc.filed_date)
        })
        """
        tx.run(query, docs=docs)
    
    def create_relationships(self, claims_df: pd.DataFrame, batch_size: int = 5000):
        """Create relationships between nodes"""
        logger.info("Creating relationships...")
        
        total_created = 0
        for i in range(0, len(claims_df), batch_size):
            batch = claims_df.iloc[i:i+batch_size]
            edges = batch[['claim_id', 'claimant_id', 'policy_id', 'city']].to_dict('records')
            
            with self.driver.session() as session:
                session.execute_write(self._create_edges_batch, edges)
            
            total_created += len(edges)
            if total_created % 10000 == 0:
                logger.info(f"  Created relationships for {total_created:,} claims")
        
        logger.success(f"✓ Created relationships for {total_created:,} claims")
    
    @staticmethod
    def _create_edges_batch(tx, edges: List[Dict]):
        query = """
        UNWIND $edges AS edge
        MATCH (c:Claim {claim_id: edge.claim_id})
        
        MERGE (p:Claimant {claimant_id: edge.claimant_id})
        CREATE (c)-[:FILED_BY]->(p)
        
        MERGE (pol:Policy {policy_id: edge.policy_id})
        CREATE (c)-[:ON_POLICY]->(pol)
        
        MERGE (city:City {city_name: edge.city})
        CREATE (c)-[:IN_LOCATION]->(city)
        """
        tx.run(query, edges=edges)
    
    def create_document_links(self, docs_df: pd.DataFrame):
        """Link documents to claims"""
        logger.info("Linking documents to claims...")
        
        links = docs_df[['claim_id', 'file_name', 'is_delayed_filing']].to_dict('records')
        with self.driver.session() as session:
            session.execute_write(self._create_doc_links_batch, links)
        
        logger.success(f"✓ Linked {len(links):,} documents")
    
    @staticmethod
    def _create_doc_links_batch(tx, links: List[Dict]):
        query = """
        UNWIND $links AS link
        MATCH (c:Claim {claim_id: link.claim_id})
        MATCH (d:Document {file_name: link.file_name})
        CREATE (c)-[:HAS_DOCUMENT {is_suspicious: link.is_delayed_filing}]->(d)
        """
        tx.run(query, links=links)
