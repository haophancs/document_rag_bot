from pathlib import Path
from typing import Any, Dict, List

import requests
import streamlit as st
import yaml

# Load settings for API connection
from document_rag_bot.settings import settings

API_URL = f"http://{settings.api_host}:{settings.api_port}/api"


with open(Path("./assets/ui_texts.yaml"), "r", encoding="utf-8") as file:
    ui_texts = yaml.safe_load(file)


def initialize_session_state() -> None:
    """
    Initialize session state variables if they do not exist.

    :returns: None
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "follow_up_mode" not in st.session_state:
        st.session_state.follow_up_mode = False


def display_chat_messages() -> None:
    """
    Display all messages in the chat interface from the session state.

    :returns: None
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def call_workflow_api(
    messages: List[Dict[str, Any]],
    ask_follow_up: bool = False,
) -> Dict[str, Any]:
    """
    Call the workflow API endpoint and return the generated answer.

    :param messages: List of message dictionaries containing role and content.
    :param ask_follow_up: Boolean flag to request follow-up questions.
    :returns: The generated answer from the workflow API.
    :raises requests.exceptions.RequestException: If the API call fails.
    """
    try:
        response = requests.post(
            f"{API_URL}/run-workflow",
            headers={"accept": "application/json", "Content-Type": "application/json"},
            json=messages,
            params={"ask_follow_up": ask_follow_up},
            timeout=300,
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as err:
        st.error(f"Error calling workflow API: {err}")
        return {
            "answer": "Sorry, an error occurred while processing. Please try again.",
            "references": [],
        }


def process_message(prompt: str, ask_follow_up: bool = True) -> None:
    """
    Process a user message and generate a response.

    :param prompt: The user's input message.
    :param ask_follow_up: Boolean flag to request follow-up questions.
    :returns: None
    """
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)

    # Show typing indicator while processing
    with st.chat_message("assistant"), st.spinner(
        ui_texts.get("thinking_message", "Thinking..."),
    ):
        # Call workflow API
        response = call_workflow_api(st.session_state.messages, ask_follow_up)

        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": response["answer"]},
        )

        # Display assistant response
        st.write(response["answer"])

        # Display references if available
        if response.get("references"):
            st.write("**Reference(s):**")
            for reference in response["references"]:
                st.write(str(reference["indices"]), reference["url"])


def main() -> None:
    """
    Main function to run the chatbot application.

    :returns: None
    """
    # Set page configuration
    st.set_page_config(
        page_title=ui_texts.get("app_title", "AI Assistant"),
        page_icon="🤖",
    )

    # Set the app title
    st.title(ui_texts.get("app_title", "AI Assistant"))

    with st.spinner("Waiting for API..."):
        while True:
            try:
                requests.get(f"{API_URL}/health")  # noqa: S113
                break
            except requests.exceptions.ConnectionError:
                pass

    # Initialize session state
    initialize_session_state()

    # Display chat messages
    display_chat_messages()

    # Add toggle for follow-up questions in sidebar
    st.sidebar.title(ui_texts.get("sidebar_title", "Chat Settings"))
    enable_follow_up = st.sidebar.checkbox(
        "Follow-up Asking",
        value=False,
        help=(
            "[EXPERIMENTAL] When enabled, the AI will consider all conversation history"
        ),
    )

    # Chat input
    if prompt := st.chat_input(
        ui_texts.get("input_placeholder", "How can I help you today?"),
    ):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        process_message(prompt, ask_follow_up=enable_follow_up)

    # Add a button to clear chat history
    if st.sidebar.button(
        ui_texts.get("clear_history_button", "Clear Chat History"),
    ):
        st.session_state.messages = []
        st.rerun()

    # Display chat information in the sidebar
    st.sidebar.info(
        ui_texts.get(
            "sidebar_info",
            "Messages in conversation: {messages_count}",
        ).format(
            messages_count=len(st.session_state.messages),
        ),
    )


if __name__ == "__main__":
    main()
