import streamlit as st
import requests

# The Orchestrator URL (using the Docker internal network name)
ORCHESTRATOR_URL = "http://orchestrator:8000/chat"
EMBEDDING_BOT_URL = "http://embedding-bot:8000/sync-schemas"

st.set_page_config(page_title="RiskNexus Assistant", page_icon="🤖")

with st.sidebar:
    st.header("Admin Panel")
    if st.button("🔄 Sync DB Schemas to Vector Store"):
        with st.spinner("Syncing schemas..."):
            try:
                res = requests.post(EMBEDDING_BOT_URL)
                res.raise_for_status()
                st.success("Schemas synced successfully!")
            except Exception as e:
                st.error(f"Failed to sync schemas: {e}")
                
st.title("RiskNexus Agentic RAG")

# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display all previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Optionally display citations if the bot provided them
        if "citations" in message and message["citations"]:
            with st.expander("Sources"):
                for cite in message["citations"]:
                    st.write(f"- {cite}")

# Handle user input
if prompt := st.chat_input("Ask RiskNexus a question..."):
    # 1. Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Prepare payload for the Orchestrator
    # We send the current question AND the full chat history for context!
    payload = {
        "query": prompt,
        "context": {
            # Let the Orchestrator default to 'documents' or 'regulations' as needed,
            # or you can add a dropdown in Streamlit to let the user select the collection.
            "collection": "documents", 
            "chat_history": st.session_state.messages # Passing the memory!
        }
    }

    # 3. Call the Orchestrator
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(ORCHESTRATOR_URL, json=payload)
                response.raise_for_status()
                data = response.json()
                
                # Extract the final answer and citations
                # Извличаме финалното обобщение от Orchestrator-а
                answer = data.get("final_summary", "No answer provided.")
                
                # Събираме всички цитати от всички извикани ботове (worker_responses)
                citations = []
                worker_responses = data.get("worker_responses", [])
                for worker_data in worker_responses:
                    if "citations" in worker_data and worker_data["citations"]:
                        citations.extend(worker_data["citations"])
                
                # Премахваме дублиращите се цитати
                citations = list(set(citations))

                # Display the answer
                st.markdown(answer)
                if citations:
                    with st.expander("Sources"):
                        for cite in citations:
                            st.write(f"- {cite}")

                # 4. Save assistant response to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "citations": citations
                })

            except requests.exceptions.RequestException as e:
                st.error(f"Error communicating with Orchestrator: {e}")