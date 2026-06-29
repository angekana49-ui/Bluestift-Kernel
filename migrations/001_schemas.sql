-- 001_schemas.sql
-- Create the logical schemas used across Bluestift.
-- Run order: 001 -> 002 -> 003 -> 004.

CREATE SCHEMA IF NOT EXISTS kernel;
CREATE SCHEMA IF NOT EXISTS learning;
CREATE SCHEMA IF NOT EXISTS schools;
CREATE SCHEMA IF NOT EXISTS rag;
CREATE SCHEMA IF NOT EXISTS content;
