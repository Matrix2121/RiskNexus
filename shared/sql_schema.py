class RiskSQLSchema:
    def __init__(self):
        # Define tables and their available columns
        self.tables = {
            "clients": [
                "client_id (PK)", "client_type", "name", "registration_number", 
                "industry", "country", "risk_rating", "created_at"
            ],
            "financial_reports": [
                "report_id (PK)", "client_id (FK)", "fiscal_year", "revenue", 
                "profit", "current_assets", "current_liabilities", "equity", "credit_score"
            ],
            "loans": [
                "loan_id (PK)", "principal_amount", "outstanding_balance", 
                "currency", "interest_rate", "status", "issue_date", "maturity_date"
            ],
            "sanctions": [
                "sanction_id (PK)", "source", "reason", "imposed_date", "is_active"
            ]
        }
        
        # Explicitly define relationships for JOINs
        self.relations = [
            "financial_reports.client_id = clients.client_id"
        ]
        
        # Contextual notes to help the LLM understand the data mapping
        self.notes = [
            "client_id maps to Neo4j Company.id and Person.id.",
            "loan_id maps to Neo4j Loan.id.",
            "sanction_id maps to Neo4j Sanction.id.",
            "Use standard PostgreSQL syntax.",
            "Always use table aliases when performing JOINs."
        ]

    def get_sql_context(self) -> str:
        """Returns a formatted string of the SQL schema for the Gemini prompt."""
        schema_parts = ["Database Schema:"]
        
        for table, columns in self.tables.items():
            schema_parts.append(f"Table '{table}' columns: {', '.join(columns)}")
            
        schema_parts.append(f"Foreign Key Relationships: {', '.join(self.relations)}")
        schema_parts.append(f"Notes: {' '.join(self.notes)}")
        
        return "\n".join(schema_parts)