-- 008_kernel_monitoring_alerts.sql
-- Enrich kernel_monitoring with the pedagogical-safety alert schema
-- (Hooshyar/Bastani/Amodei). Additive: existing generic log rows keep working.
ALTER TABLE kernel.kernel_monitoring ADD COLUMN IF NOT EXISTS user_id uuid;
ALTER TABLE kernel.kernel_monitoring ADD COLUMN IF NOT EXISTS concept_id uuid;
ALTER TABLE kernel.kernel_monitoring ADD COLUMN IF NOT EXISTS alert_type text;
ALTER TABLE kernel.kernel_monitoring ADD COLUMN IF NOT EXISTS alert_severity text;
ALTER TABLE kernel.kernel_monitoring ADD COLUMN IF NOT EXISTS alert_details jsonb DEFAULT '{}'::jsonb;
ALTER TABLE kernel.kernel_monitoring ADD COLUMN IF NOT EXISTS inconsistency_rate float;
ALTER TABLE kernel.kernel_monitoring ADD COLUMN IF NOT EXISTS volatility_score float;
ALTER TABLE kernel.kernel_monitoring ADD COLUMN IF NOT EXISTS resolved boolean DEFAULT false;
ALTER TABLE kernel.kernel_monitoring ADD COLUMN IF NOT EXISTS resolved_by text;
ALTER TABLE kernel.kernel_monitoring ADD COLUMN IF NOT EXISTS resolved_at timestamptz;
ALTER TABLE kernel.kernel_monitoring ADD COLUMN IF NOT EXISTS window_start timestamptz;
ALTER TABLE kernel.kernel_monitoring ADD COLUMN IF NOT EXISTS window_end timestamptz;
ALTER TABLE kernel.kernel_monitoring ADD COLUMN IF NOT EXISTS interactions_count int;

ALTER TABLE kernel.kernel_monitoring DROP CONSTRAINT IF EXISTS kernel_monitoring_alert_type_chk;
ALTER TABLE kernel.kernel_monitoring ADD CONSTRAINT kernel_monitoring_alert_type_chk
  CHECK (alert_type IS NULL OR alert_type IN (
    'passive_dependency','false_mastery','re_emergence_error','cognitive_overload',
    'fixed_mindset','ood_distribution','inconsistency_high'));

ALTER TABLE kernel.kernel_monitoring DROP CONSTRAINT IF EXISTS kernel_monitoring_alert_severity_chk;
ALTER TABLE kernel.kernel_monitoring ADD CONSTRAINT kernel_monitoring_alert_severity_chk
  CHECK (alert_severity IS NULL OR alert_severity IN ('low','medium','high'));

CREATE INDEX IF NOT EXISTS kernel_monitoring_user_idx ON kernel.kernel_monitoring(user_id);
CREATE INDEX IF NOT EXISTS kernel_monitoring_alert_idx ON kernel.kernel_monitoring(alert_type, resolved);
