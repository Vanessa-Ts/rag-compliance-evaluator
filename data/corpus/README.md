# Corpus doc_id Registry

This directory holds the ground-truth document corpus for the multi-jurisdiction
employment-law Q&A RAG service. Each markdown file is a concise, factual summary
(~300–600 words) of a real employment-law topic and begins with YAML frontmatter
containing exactly four keys: `doc_id`, `title`, `jurisdiction`, `source_url`.

This README is the **authoritative doc_id registry**. The golden evaluation set
(WP7) validates retrieval against the `doc_id` values listed here, so this list
must stay accurate and complete. When adding or renaming a document, update this
table in the same change.

These documents are factual summaries, not legal advice.

## Registry

| doc_id | jurisdiction | topic |
|---|---|---|
| `eu-working-time-directive` | EU | Working Time Directive 2003/88/EC — max 48-hour week, daily/weekly rest, breaks, ≥4 weeks paid leave, night work. |
| `gdpr-employee-data-basics` | EU | GDPR (Reg. 2016/679) applied to employee data — lawful bases, principles, employee rights, workplace monitoring, breaches. |
| `de-labor-law-overview` | DE | German labor law basics — working hours (ArbZG), statutory vacation (BUrlG), notice and dismissal protection (KSchG), works councils, sick pay. |
| `fr-labor-law-overview` | FR | French labor law basics — 35-hour week and overtime, CDI/CDD contracts, 5 weeks paid leave, dismissal cause and severance, SMIC. |
| `es-labor-law-overview` | ES | Spanish labor law basics — Workers' Statute contracts, 40-hour week, severance scales (20/33 days), holidays, SMI. |
| `nl-labor-law-overview` | NL | Dutch labor law basics — fixed-term chain rule, working hours, holiday + 8% allowance, service-based notice, transition payment. |
| `eu-parental-leave` | EU | Work-Life Balance Directive (EU) 2019/1158 — paternity, parental (4 months, 2 non-transferable), carers' leave, flexible working. |

7 documents total.
