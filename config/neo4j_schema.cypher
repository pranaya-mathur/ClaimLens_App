// ClaimLens Neo4j Schema
// Run this in Neo4j Browser after starting the database

// ============================================================
// CREATE CONSTRAINTS (Uniqueness)
// ============================================================

CREATE CONSTRAINT claim_id_unique IF NOT EXISTS
FOR (c:Claim) REQUIRE c.claim_id IS UNIQUE;

CREATE CONSTRAINT claimant_id_unique IF NOT EXISTS
FOR (p:Claimant) REQUIRE p.claimant_id IS UNIQUE;

CREATE CONSTRAINT policy_id_unique IF NOT EXISTS
FOR (pol:Policy) REQUIRE pol.policy_id IS UNIQUE;

CREATE CONSTRAINT document_name_unique IF NOT EXISTS
FOR (d:Document) REQUIRE d.file_name IS UNIQUE;

CREATE CONSTRAINT city_name_unique IF NOT EXISTS
FOR (c:City) REQUIRE c.city_name IS UNIQUE;

// ============================================================
// CREATE INDEXES (Performance)
// ============================================================

CREATE INDEX claim_fraud_score IF NOT EXISTS
FOR (c:Claim) ON (c.fraud_score);

CREATE INDEX claim_amount IF NOT EXISTS
FOR (c:Claim) ON (c.amount);

CREATE INDEX claim_product IF NOT EXISTS
FOR (c:Claim) ON (c.product);

CREATE INDEX claimant_fraud_rate IF NOT EXISTS
FOR (p:Claimant) ON (p.fraud_rate);

CREATE INDEX document_type IF NOT EXISTS
FOR (d:Document) ON (d.doc_type);

CREATE INDEX document_delay IF NOT EXISTS
FOR (d:Document) ON (d.filing_delay);

// ============================================================
// SAMPLE QUERIES (for testing)
// ============================================================

// 1. Count nodes by type
MATCH (n)
RETURN labels(n)[0] AS type, COUNT(n) AS count
ORDER BY count DESC;

// 2. Find high fraud score claims
MATCH (c:Claim)
WHERE c.fraud_score > 0.8
RETURN c.claim_id, c.amount, c.fraud_score
ORDER BY c.fraud_score DESC
LIMIT 10;

// 3. Find fraud rings (shared documents)
MATCH (c1:Claim)-[:HAS_DOCUMENT]->(d:Document)<-[:HAS_DOCUMENT]-(c2:Claim)
WHERE c1.claim_id < c2.claim_id
MATCH (c1)-[:FILED_BY]->(p1:Claimant)
MATCH (c2)-[:FILED_BY]->(p2:Claimant)
WHERE p1.claimant_id <> p2.claimant_id
WITH p1, p2, COUNT(DISTINCT d) AS shared_docs
WHERE shared_docs >= 2
RETURN p1.claimant_id, p2.claimant_id, shared_docs
ORDER BY shared_docs DESC
LIMIT 20;