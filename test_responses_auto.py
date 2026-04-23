
import os
from responses import ask_with_metadata

query = "Give me a comparative table for FY 2024-25 for following companies: IndiGo Airlines, Indigo Paints, Apollo Hospitals Delhi, Oracle Financial Services, Data Patterns, KPEL. I need: Total Assets, Revenue from operations, Net Profit, and Number of Employees."

print(f"Running test query: {query}")
result = ask_with_metadata(query)

print("\n--- ANSWER ---\n")
print(result["answer"])
print("\n--- METADATA ---\n")
print(f"Time Taken: {result['time_taken']}")
print(f"Chunks Count: {result['chunks_count']}")
