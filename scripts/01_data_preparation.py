"""
Data Preparation Script
Prepare raw CSVs for graph loading
"""
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger


def prepare_claims_data():
    """Enrich claims data with derived features"""
    logger.info("Loading raw claims data...")
    
    # Load main claims
    claims = pd.read_csv('data/raw/claimlens_production_50k.csv')
    
    logger.info(f"Loaded {len(claims):,} claims")
    
    # Convert date
    claims['incident_date'] = pd.to_datetime(claims['incident_date'])
    
    # Derived features
    claims['is_high_value'] = claims['claim_amount'] > claims['claim_amount'].quantile(0.75)
    claims['is_new_policy'] = claims['days_since_policy_start'] <= 30
    claims['has_red_flags'] = claims['red_flags'].apply(
        lambda x: x not in ['none', 'None', '', '"none"'] if pd.notna(x) else False
    )
    
    # Save enriched
    output_path = 'data/processed/claims_enriched.csv'
    claims.to_csv(output_path, index=False)
    logger.success(f"✓ Saved: {output_path}")
    
    return claims


def prepare_claimant_profiles(claims_df):
    """Aggregate claimant-level profiles"""
    logger.info("Creating claimant profiles...")
    
    profiles = claims_df.groupby('claimant_id').agg({
        'claim_id': 'count',
        'claim_amount': ['sum', 'mean', 'max'],
        'fraud_label': lambda x: (x == 1).sum(),
        'fraud_score': 'mean',
        'incident_date': ['min', 'max']
    }).reset_index()
    
    profiles.columns = [
        'claimant_id', 'total_claims', 'total_claimed_amount',
        'avg_claim_amount', 'max_claim_amount', 'fraud_claim_count',
        'avg_fraud_score', 'first_claim_date', 'last_claim_date'
    ]
    
    profiles['fraud_rate'] = profiles['fraud_claim_count'] / profiles['total_claims']
    profiles['is_repeat_claimant'] = profiles['total_claims'] >= 3
    profiles['is_high_risk'] = (profiles['fraud_rate'] >= 0.5) & (profiles['total_claims'] >= 2)
    
    output_path = 'data/processed/claimant_profiles.csv'
    profiles.to_csv(output_path, index=False)
    logger.success(f"✓ Saved: {output_path} ({len(profiles):,} claimants)")
    
    return profiles


def prepare_policy_profiles(claims_df):
    """Aggregate policy-level data"""
    logger.info("Creating policy profiles...")
    
    profiles = claims_df.groupby('policy_id').agg({
        'claim_id': 'count',
        'claimant_id': 'nunique',
        'claim_amount': 'sum',
        'fraud_label': lambda x: (x == 1).sum()
    }).reset_index()
    
    profiles.columns = [
        'policy_id', 'claim_count', 'unique_claimants',
        'total_amount', 'fraud_count'
    ]
    
    profiles['has_multiple_claimants'] = profiles['unique_claimants'] > 1
    
    output_path = 'data/processed/policy_profiles.csv'
    profiles.to_csv(output_path, index=False)
    logger.success(f"✓ Saved: {output_path} ({len(profiles):,} policies)")
    
    return profiles


def prepare_documents():
    """Prepare document metadata"""
    logger.info("Processing documents...")
    
    docs = pd.read_csv('data/raw/synthetic_docs_metadata_500.csv')
    
    # Calculate delays
    docs['incident_date'] = pd.to_datetime(docs['incident_date'])
    docs['filed_date'] = pd.to_datetime(docs['filed_date'])
    docs['filing_delay_days'] = (docs['filed_date'] - docs['incident_date']).dt.days
    
    docs['is_delayed_filing'] = docs['filing_delay_days'] > 90
    docs['is_extreme_delay'] = docs['filing_delay_days'] > 365
    
    output_path = 'data/processed/documents_enriched.csv'
    docs.to_csv(output_path, index=False)
    logger.success(f"✓ Saved: {output_path} ({len(docs):,} documents)")
    
    return docs


def main():
    """Main data preparation pipeline"""
    logger.info("="*60)
    logger.info("CLAIMLENS DATA PREPARATION")
    logger.info("="*60)
    
    # Create output directory
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    claims = prepare_claims_data()
    claimants = prepare_claimant_profiles(claims)
    policies = prepare_policy_profiles(claims)
    docs = prepare_documents()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Claims: {len(claims):,}")
    logger.info(f"Claimants: {len(claimants):,}")
    logger.info(f"  - Repeat claimants (3+): {claimants['is_repeat_claimant'].sum():,}")
    logger.info(f"  - High-risk claimants: {claimants['is_high_risk'].sum():,}")
    logger.info(f"Policies: {len(policies):,}")
    logger.info(f"  - Multi-claimant policies: {policies['has_multiple_claimants'].sum():,}")
    logger.info(f"Documents: {len(docs):,}")
    logger.info(f"  - Delayed filings: {docs['is_delayed_filing'].sum():,}")
    logger.success("\n✓ Data preparation complete!")


if __name__ == "__main__":
    main()
