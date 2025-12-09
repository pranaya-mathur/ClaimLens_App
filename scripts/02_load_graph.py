"""
Load Fraud Graph into Neo4j
"""
import pandas as pd
from loguru import logger
import sys
sys.path.append('.')

from src.fraud_engine.graph_loader import FraudGraphLoader
from config.settings import get_settings


def main():
    """Load all data into Neo4j"""
    logger.info("="*60)
    logger.info("LOADING FRAUD GRAPH INTO NEO4J")
    logger.info("="*60)
    
    settings = get_settings()
    
    # Initialize loader
    logger.info(f"Connecting to Neo4j at {settings.NEO4J_URI}...")
    loader = FraudGraphLoader(
        uri=settings.NEO4J_URI,
        user=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD
    )
    
    try:
        # Load data
        logger.info("\nStep 1: Loading claims...")
        claims = pd.read_csv('data/processed/claims_enriched.csv')
        loader.load_claims(claims)
        
        logger.info("\nStep 2: Loading claimants...")
        claimants = pd.read_csv('data/processed/claimant_profiles.csv')
        loader.load_claimants(claimants)
        
        logger.info("\nStep 3: Loading policies...")
        policies = pd.read_csv('data/processed/policy_profiles.csv')
        loader.load_policies(policies)
        
        logger.info("\nStep 4: Loading documents...")
        docs = pd.read_csv('data/processed/documents_enriched.csv')
        loader.load_documents(docs)
        
        logger.info("\nStep 5: Creating relationships...")
        loader.create_relationships(claims)
        
        logger.info("\nStep 6: Linking documents...")
        loader.create_document_links(docs)
        
        logger.success("\n" + "="*60)
        logger.success("✓ FRAUD GRAPH LOADED SUCCESSFULLY!")
        logger.success("="*60)
        logger.info(f"\nAccess Neo4j Browser: http://localhost:7474")
        logger.info(f"Username: {settings.NEO4J_USER}")
        logger.info(f"Password: {settings.NEO4J_PASSWORD}")
    
    finally:
        loader.close()


if __name__ == "__main__":
    main()
