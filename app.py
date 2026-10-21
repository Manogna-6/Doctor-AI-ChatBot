import os
import gradio as gr
from google import genai
from google.genai import types

# =========================================================
# 1. GEMINI CONFIGURATION
# =========================================================

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "Gemini API key not found. "
        "Set the GEMINI_API_KEY environment variable first."
    )

client = genai.Client(api_key=API_KEY)

# Keep your existing model name.
MODEL_NAME = "gemini-3.8-flash"


# =========================================================
# 2. MEDICAL SAFETY INSTRUCTIONS
# =========================================================

SYSTEM_INSTRUCTION = """
You are AI Medical Chatbot, a friendly AI health information assistant.

Your responsibilities:
1. Explain common symptoms in simple language.
2. Ask relevant follow-up questions when necessary.
3. Explain possible causes without claiming a definite diagnosis.
4. Offer general self-care and wellness information.
5. Explain when the user should seek professional medical care.
6. Use previous conversation messages to understand follow-up questions.
7. Be empathetic, calm, clear, and concise.

IMPORTANT SAFETY RULES:
- You are not a doctor and cannot diagnose medical conditions.
- Do not claim certainty about diseases based only on symptoms.
- Do not prescribe medicines or provide individualized medication doses.
- Do not estimate life expectancy from symptoms.
- For serious or worsening symptoms, recommend prompt professional care.
- If the user reports chest pain, severe difficulty breathing,
  signs of stroke, loss of consciousness, or another possible
  medical emergency, tell them to seek emergency medical help
  immediately.
- Do not delay emergency care by asking more questions.
- Ask the user's age and symptom duration when relevant.
- Clearly distinguish general health information from medical advice.
- Never pretend to have examined the user.
"""


# =========================================================
# 3. CONVERT CHAT HISTORY FOR GEMINI
# =========================================================

def build_gemini_history(history):
    contents = []

    # Keep the most recent 12 history entries.
    for item in history[-12:]:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        message = item.get("content", "")

        # Skip non-text messages.
        if not isinstance(message, str):
            continue

        if not message.strip():
            continue

        if role == "user":
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(text=message)
                    ]
                )
            )

        elif role == "assistant":
            contents.append(
                types.Content(
                    role="model",
                    parts=[
                        types.Part(text=message)
                    ]
                )
            )

    return contents


# =========================================================
# 4. GEMINI CHAT FUNCTION
# =========================================================

def chat_bot(message, history):
    history = history or []

    # Validate the input.
    if not message or not message.strip():
        return (
            history,
            "",
            "Please enter your question.",
            "Enter a question to get started."
        )

    try:
        # Convert previous messages.
        contents = build_gemini_history(history)

        # Add the current user message.
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part(text=message.strip())
                ]
            )
        )

        print("Calling Gemini model:", MODEL_NAME)

        # Call Gemini using your existing client setup.
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.4,
                max_output_tokens=900
            )
        )

        answer = response.text

        if not answer:
            raise ValueError(
                "Gemini returned an empty response."
            )

        # Update chat history only after a successful response.
        updated_history = history + [
            {
                "role": "user",
                "content": message.strip()
            },
            {
                "role": "assistant",
                "content": answer
            }
        ]

        return (
            updated_history,
            "",
            "Response generated successfully.",
            answer
        )

    except Exception as e:

        # Print technical details in the terminal for debugging.
        import traceback
        traceback.print_exc()

        error_message = str(e)
        error_upper = error_message.upper()

        # -------------------------------------------------
        # HANDLE GEMINI 503 UNAVAILABLE
        # -------------------------------------------------

        if (
            "503" in error_message
            or "UNAVAILABLE" in error_upper
        ):
            status = "Gemini is temporarily busy."

            friendly_answer = (
                "### 🐱 AI Medical Chatbot is temporarily busy\n\n"
                "The Gemini service is experiencing high demand. "
                "Please wait 30–60 seconds and try again.\n\n"
                "Your question has not been answered yet. "
                "Please try sending it again later."
            )

        # -------------------------------------------------
        # HANDLE GEMINI 429 RESOURCE EXHAUSTED
        # -------------------------------------------------

        elif (
            "429" in error_message
            or "RESOURCE_EXHAUSTED" in error_upper
        ):
            status = "Gemini API usage limit reached."

            friendly_answer = (
                "### ⏳ Temporary API limit\n\n"
                "The Gemini API usage limit may have been reached.\n\n"
                "Please check the quota and rate limits associated "
                "with your Gemini API key. Wait before trying again."
            )

        # -------------------------------------------------
        # HANDLE API KEY OR PERMISSION ERRORS
        # -------------------------------------------------

        elif (
            "401" in error_message
            or "403" in error_message
            or "PERMISSION_DENIED" in error_upper
            or "UNAUTHENTICATED" in error_upper
        ):
            status = "Gemini API authentication or permission error."

            friendly_answer = (
                "### 🔑 API key or permission problem\n\n"
                "Please check that your GEMINI_API_KEY environment "
                "variable is configured correctly and that your "
                "API key has permission to access the selected model.\n\n"
                "Do not share your API key."
            )

        # -------------------------------------------------
        # HANDLE MODEL NOT FOUND
        # -------------------------------------------------

        elif (
            "404" in error_message
            or "NOT_FOUND" in error_upper
        ):
            status = "Gemini model not found or unavailable."

            friendly_answer = (
                "### ⚙️ Model unavailable\n\n"
                "The configured Gemini model may not be available "
                "to your API key.\n\n"
                "Check the model name and verify that your Gemini "
                "API account supports it."
            )

        # -------------------------------------------------
        # HANDLE OTHER ERRORS
        # -------------------------------------------------

        else:
            status = "Could not get a response from Gemini."

            friendly_answer = (
                "### ⚠️ Something went wrong\n\n"
                "AI Medical Chatbot could not process your request.\n\n"
                "Please try again later. If the problem continues, "
                "check the terminal for the technical error."
            )

        # Log the technical error locally.
        print("Gemini API error:", error_message)

        # Return a friendly error without destroying chat history.
        return (
            history,
            "",
            status,
            friendly_answer
        )


# =========================================================
# 5. CLEAR CHAT
# =========================================================

def clear_chat():
    return (
        [],
        "",
        "Conversation cleared.",
        "Ask AI Medical Chatbot a health-related question to get started."
    )


# =========================================================
# 6. QUICK QUESTION HELPER
# =========================================================

def set_question(question):
    return question


# =========================================================
# 7. CUSTOM CSS
# =========================================================

custom_css = """
.gradio-container {
    max-width: 1150px !important;
    margin: auto !important;
}

#main-title {
    text-align: center;
    padding: 22px;
    border-radius: 18px;
    background: linear-gradient(120deg, #e0f2fe, #dcfce7);
    margin-bottom: 18px;
}

#main-title h1 {
    color: #075985;
    font-size: 34px;
    margin-bottom: 8px;
}

#main-title p {
    color: #334155;
    font-size: 16px;
}

#chat-panel,
#advice-panel {
    border-radius: 16px;
    border: 1px solid #dbeafe;
    padding: 12px;
}

#status-box {
    border-radius: 10px;
    padding: 8px;
}

.quick-btn {
    border-radius: 10px !important;
}

footer {
    visibility: hidden;
}
"""


# =========================================================
# 8. BUILD THE GRADIO USER INTERFACE
# =========================================================

with gr.Blocks(title="AI Medical Chatbot") as demo:

    # Application header.
    gr.HTML(
        """
        <div id="main-title">
            <h1>🐱 AI Medical Chatbot</h1>
            <p>Your friendly AI health information assistant</p>
            <p>
                Ask questions about symptoms, wellness,
                and general health.
            </p>
        </div>
        """
    )

    with gr.Row():

        # =============================================
        # LEFT PANEL: CHAT INTERFACE
        # =============================================

        with gr.Column(scale=3):

            with gr.Group(elem_id="chat-panel"):

                chatbot = gr.Chatbot(
                    label="Chat with AI Medical Chatbot",
                    height=440
                )

                message_input = gr.Textbox(
                    label="Your question",
                    placeholder=(
                        "Example: I've had a headache since yesterday..."
                    ),
                    lines=2
                )

                with gr.Row():

                    send_button = gr.Button(
                        "💬 Ask AI Medical Chatbot",
                        variant="primary"
                    )

                    clear_button = gr.Button(
                        "🗑️ Clear conversation"
                    )

            # Suggested questions.
            gr.Markdown("### 💡 Suggested questions")

            with gr.Row():

                fever_button = gr.Button(
                    "🤒 Fever",
                    elem_classes=["quick-btn"]
                )

                headache_button = gr.Button(
                    "🤕 Headache",
                    elem_classes=["quick-btn"]
                )

            with gr.Row():

                cough_button = gr.Button(
                    "😷 Cough",
                    elem_classes=["quick-btn"]
                )

                sleep_button = gr.Button(
                    "😴 Sleep problems",
                    elem_classes=["quick-btn"]
                )

        # =============================================
        # RIGHT PANEL: HEALTH INFORMATION
        # =============================================

        with gr.Column(scale=2):

            with gr.Group(elem_id="advice-panel"):

                gr.Markdown("## 🩺 Health Information")

                advice_output = gr.Markdown(
                    value=(
                        "Your health information will appear here "
                        "after you ask a question."
                    )
                )

                status_output = gr.Textbox(
                    label="Status",
                    value="AI Medical Chatbot is ready.",
                    interactive=False,
                    elem_id="status-box"
                )

            gr.Markdown(
                """
                ### 🛡️ Health and safety

                - Share only information needed for your question.
                - Avoid entering names, addresses, or other identifying details.
                - Seek professional care for persistent or worsening symptoms.
                - Never delay emergency treatment to use this chatbot.
                """
            )

    # =====================================================
    # 9. CONNECT THE SEND BUTTON
    # =====================================================

    send_button.click(
        fn=chat_bot,
        inputs=[
            message_input,
            chatbot
        ],
        outputs=[
            chatbot,
            message_input,
            status_output,
            advice_output
        ]
    )

    # Pressing Enter submits the question.
    message_input.submit(
        fn=chat_bot,
        inputs=[
            message_input,
            chatbot
        ],
        outputs=[
            chatbot,
            message_input,
            status_output,
            advice_output
        ]
    )

    # =====================================================
    # 10. CLEAR THE CONVERSATION
    # =====================================================

    clear_button.click(
        fn=clear_chat,
        inputs=[],
        outputs=[
            chatbot,
            message_input,
            status_output,
            advice_output
        ]
    )

    # =====================================================
    # 11. SUGGESTED QUESTION BUTTONS
    # =====================================================

    fever_button.click(
        fn=lambda: set_question(
            "What should I know about fever, and when should I seek medical care?"
        ),
        outputs=message_input
    )

    headache_button.click(
        fn=lambda: set_question(
            "What are common causes of headaches, and what warning signs should I watch for?"
        ),
        outputs=message_input
    )

    cough_button.click(
        fn=lambda: set_question(
            "What are common causes of a cough, and when should I consult a doctor?"
        ),
        outputs=message_input
    )

    sleep_button.click(
        fn=lambda: set_question(
            "What are some general ways to improve sleep?"
        ),
        outputs=message_input
    )


# =========================================================
# 12. LAUNCH THE APPLICATION
# =========================================================

if __name__ == "__main__":

    demo.launch(
        share=True,
        theme=gr.themes.Soft(),
        css=custom_css
    )