class RiskGraphSchema:
    def __init__(self):
        self.node_labels = [
            "Company",              #указва уридическо лице
            "Person",               #указва физическо лице
            "Loan",                 #указва кридит
            "Sanction"              #указва санкции
        ]
        self.relationship_types = [
            "OWNER_OF",             #указва собственост
            "REPRESENTED_BY",       #указва управители
            "BENEFICIAL_OWNER",     #указва бенефициент
            "OBLIGOR_OF",           #указва главен длъжник по кредит
            "GUARANTOR_OF",         #указва гарант по кредит
            "SUBJECT_TO",           #указва връзка към санкции
            "SUPPLIES",             #указва търгосвка зависимост
            "FRANCHISE_OF"          #указва специфична търговска зависимост
        ]
        self.constraints = [
            "Company(id) - UNIQUE, REQUIRED",
            "Person(id) - UNIQUE, REQUIRED",
            "Loan(id) - UNIQUE, REQUIRED",
            "Sanction(id) - UNIQUE, REQUIRED",
            "Company(name) - INDEXED",
            "Person(name) - INDEXED"
        ]
        
    def get_cypher_context(self) -> str:
        """Връща описание на схемата за промпта на Gemini."""
        return (
            f"Възли: {', '.join(self.node_labels)}. "
            f"Връзки: {', '.join(self.relationship_types)}. "
            "Важно: Company.id и Person.id са свързани с ClientID в SQL базата."
            "Важно: Loan.id е свързан с LoanID в SQL базата."
        )