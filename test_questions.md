# RAG Test Questions

## Simple (5 chunks should work)
1. What is the total revenue of Apollo Hospitals for FY 2024-25?
2. Who is the CEO of IndiGo Airlines?
3. What dividend did Oracle Financial Services declare for FY 2024-25?

## Medium (10-15 chunks needed)
4. Compare the revenue and net profit of Apollo Hospitals and IndiGo Airlines for FY 2024-25.
5. What are the key risks mentioned in the Data Patterns and KPEL annual reports?
6. Summarize the CSR activities of Indigo Paints and Apollo Hospitals.

## Hard — Exposes RAG Limitations (20-30+ chunks needed)
7. Compare the revenue, net profit, total assets, debt-to-equity ratio, and across all six companies — Apollo Hospitals, Data Patterns, IndiGo Airlines, Indigo Paints, KPEL, and Oracle Financial Services — for FY 2024-25. Also summarize each company's key risks, future outlook, and dividend policy.
8. **Structured Analysis:** Create a comparative financial and strategic analysis table for FY 2024-25 for Apollo Hospitals, Data Patterns, IndiGo Airlines, Indigo Paints, KPEL, and Oracle Financial Services. For each company, extract precisely these values from their Consolidated Financial Statements: Revenue from Operations (in ₹ Cr), Net Profit / Profit After Tax (in ₹ Cr), Total Assets (as of March 31, 2025), Debt-to-Equity Ratio, and Total Employee Count. Additionally, summarize in bullet points: Key Risks, Future Outlook, and Dividend Policy/Declared.
9. What are the board composition details, committee structures, CSR spending, related party transactions, and auditor observations for each of the six companies?
9. Summarize the management discussion and analysis section of all six companies, including industry outlook, segment-wise revenue breakdown, capital expenditure plans, and operational challenges.
10. List all the legal proceedings, contingent liabilities, and regulatory compliance issues mentioned across all six annual reports, along with the financial impact of each.

## Edge Cases
11. What is the market cap of Tesla? (not in any document — should say "not found")
12. How did COVID-19 impact all six companies? (might not be in FY 2024-25 reports)

## aragque
13. You are analyzing a disability benefits case file for Julia Neumann. The case includes multiple documents from different sources (employer, medical provider, legal counsel, personal statement, and phone notes).
   
    Please provide a comprehensive case analysis by extracting specific information from the relevant documents:

    1) *Employment History & Status*: What is Julia Neumann's employment timeline, current position, workload percentage, and any accommodations made by her employer? (Focus on employer-related documents)

    2) *Medical Diagnosis & Severity*: What is the specific medical diagnosis, current disability rating (DoD), clinical findings, and medical recommendations regarding work capacity? (Focus on medical documentation)

    3) *Legal Case Information*: What is the case number, assigned legal counsel, current status of proceedings, and any legal recommendations? (Focus on legal/procedural documents)

    4) *Financial & Personal Circumstances*: What is Julia's monthly income, family situation (marital status, children), and how does her condition impact her personal and family life? (Focus on personal statements and communications)

    5) *Symptom Timeline & Daily Impact*: When are symptoms most severe during the day, what specific activities are affected, and how has the condition progressed over time? (Focus on documents describing daily experiences)

    6) *Cross-Document Date Verification*: Create a timeline of all key dates mentioned across documents (employment start, case filing, document submissions, etc.) and identify any date discrepancies.

    7) *Support Network & Next Steps*: What support has been provided (employer, medical, legal), what documentation is pending, and what are the recommended next actions? (Synthesize information from multiple sources)

    For each point, clearly indicate which specific document(s) contain the information and note if information is missing or contradictory across sources.
