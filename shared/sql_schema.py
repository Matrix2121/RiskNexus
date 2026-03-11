class RiskSQLSchema:
    def __init__(self):
        self.tables = {
            "clients": {
                "description": "Stores core information about corporate and individual clients.",
                "columns": {
                    "client_id (PK)": "Unique identifier for the client. Maps to Neo4j Company.id and Person.id.",
                    "client_type": "Type of client (e.g., 'Corporate', 'Individual').",
                    "name": "Full legal name of the client.",
                    "registration_number": "Official company registration number or national ID.",
                    "industry": "Economic sector or industry category.",
                    "country": "Country of registration or residence.",
                    "risk_rating": "Internal risk assessment rating (e.g., Low, Medium, High).",
                    "created_at": "Timestamp when the client record was created."
                }
            },
            "financial_reports": {
                "description": "Contains annual financial reporting data for corporate clients.",
                "columns": {
                    "report_id (PK)": "Unique identifier for the financial report.",
                    "client_id (FK)": "Reference to the client in the clients table.",
                    "fiscal_year": "The year the financial data applies to.",
                    "revenue": "Total revenue generated in the fiscal year.",
                    "profit": "Net profit or loss.",
                    "current_assets": "Value of all assets that are reasonably expected to be converted into cash within one year.",
                    "current_liabilities": "Company's short-term financial obligations.",
                    "equity": "Value of the shareholders' equity.",
                    "credit_score": "Numerical expression based on a level analysis of a person's or company's credit files."
                }
            },
            "loans": {
                "description": "Records details of loans issued to clients.",
                "columns": {
                    "loan_id (PK)": "Unique identifier for the loan. Maps to Neo4j Loan.id.",
                    "principal_amount": "The original sum of money borrowed.",
                    "outstanding_balance": "The remaining amount owed on the loan.",
                    "currency": "The currency of the loan (e.g., USD, EUR, BGN).",
                    "interest_rate": "The percentage charged on the principal by the lender.",
                    "status": "Current status of the loan (e.g., ACTIVE, CLOSED, DEFAULTED).",
                    "issue_date": "The date the loan was granted.",
                    "maturity_date": "The date the final payment is due."
                }
            },
            "sanctions": {
                "description": "Logs international or local sanctions placed on entities.",
                "columns": {
                    "sanction_id (PK)": "Unique identifier for the sanction record. Maps to Neo4j Sanction.id.",
                    "source": "The issuing authority of the sanction (e.g., OFAC, EU).",
                    "reason": "The justification for the sanction.",
                    "imposed_date": "The date the sanction took effect.",
                    "is_active": "Boolean indicating if the sanction is currently enforced."
                }
            }
        }
        self.relations = [
            "financial_reports.client_id = clients.client_id"
        ]
        self.notes = [
            "client_id maps to Neo4j Company.id and Person.id.",
            "loan_id maps to Neo4j Loan.id.",
            "sanction_id maps to Neo4j Sanction.id.",
            "Use standard PostgreSQL syntax.",
            "Always use table aliases when performing JOINs."
        ]

    def get_documents_for_embedding(self) -> list[dict]:
        """Formats the schema into individual table descriptions for vectorization."""
        docs = []
        for table_name, details in self.tables.items():
            content = f"Table: {table_name}\nDescription: {details['description']}\nColumns:\n"
            for col, desc in details["columns"].items():
                content += f"- {col}: {desc}\n"
            
            content += f"\nForeign Keys: {', '.join(self.relations)}\n"
            content += f"Important Notes: {' '.join(self.notes)}"
            
            docs.append({
                "id": f"sql_table_{table_name}",
                "text": content,
                "metadata": {"type": "sql_schema", "table_name": table_name}
            })
        return docs