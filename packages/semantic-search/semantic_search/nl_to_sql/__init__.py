"""NL-to-SQL Analytics pipeline (plan).

Translates natural-language analytics queries into Trino SQL against the
configured Athena table. Stage gates:

  schema discovery -> LLM generation -> AST security -> parallel logic
  validation -> Athena execution -> verifier-gate sufficiency check

Public surface:
- `NLToSQLPipeline.run(question, sql_hint)` returns an `AnalyticsResult`.
- All other modules are stage-internal and not part of the import surface.
"""
from semantic_search.nl_to_sql.pipeline import NLToSQLPipeline

__all__ = ['NLToSQLPipeline']
