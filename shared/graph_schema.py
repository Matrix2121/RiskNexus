class RiskGraphSchema:
    def __init__(self):
        self.nodes = {
            "Company": "Представлява юридическо лице (фирма или корпорация). Свързан е с clients таблицата в SQL чрез id.",
            "Person": "Представлява физическо лице. Може да бъде собственик, управител или свързано лице. Свързан е с clients в SQL чрез id.",
            "Loan": "Представлява кредитна експозиция. Свързан е с loans таблицата в SQL чрез id.",
            "Sanction": "Представлява наложена санкция върху лице или компания. Свързан е със sanctions таблицата в SQL чрез id."
        }
        self.relationships = {
            "OWNER_OF": "Указва собственост. Например Person е OWNER_OF Company или Company е OWNER_OF Company.",
            "REPRESENTED_BY": "Указва управители или официални представители. Company е REPRESENTED_BY Person.",
            "BENEFICIAL_OWNER": "Указва действителен собственик (UBO - Ultimate Beneficial Owner), обикновено Person е BENEFICIAL_OWNER на Company.",
            "OBLIGOR_OF": "Указва главен длъжник по кредит. Company или Person е OBLIGOR_OF Loan.",
            "GUARANTOR_OF": "Указва гарант (поръчител) по кредит. Company или Person е GUARANTOR_OF Loan.",
            "SUBJECT_TO": "Указва връзка към наложени санкции. Company или Person е SUBJECT_TO Sanction.",
            "SUPPLIES": "Указва търговска зависимост между две компании (доставчик). Company SUPPLIES Company.",
            "FRANCHISE_OF": "Указва специфична търговска франчайз зависимост. Company е FRANCHISE_OF Company."
        }

    def get_documents_for_embedding(self) -> list[dict]:
        """Formats the schema into individual node/relationship descriptions for vectorization."""
        docs = []
        
        # Embed Node definitions
        for node, desc in self.nodes.items():
            docs.append({
                "id": f"graph_node_{node}",
                "text": f"Graph Node Label: {node}\nDescription: {desc}",
                "metadata": {"type": "graph_node", "label": node}
            })
            
        # Embed Relationship definitions
        for rel, desc in self.relationships.items():
            docs.append({
                "id": f"graph_rel_{rel}",
                "text": f"Graph Relationship Type: {rel}\nDescription: {desc}",
                "metadata": {"type": "graph_relationship", "type_name": rel}
            })
            
        return docs