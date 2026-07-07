from main import State, MessageClassifier
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline


pipe = pipeline(
    "text-generation",
    model="gpt2",
    max_new_tokens=300,
)

llm = HuggingFacePipeline(pipeline=pipe)


def classify_message(state: State):
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier)

    result = classifier_llm.invoke([
        {
            "role": "system",
            "content": """Classify the user message as either:
            - 'food': if it asks for food recipes or dietary advice
            - 'gym': if it asks for gym or fitness advice
            """
        },
        {"role": "user", "content": last_message.content}
    ])
    return {"message_type": result.message_type}


def router(state: State):
    message_type = state.get("message_type", "gym")
    if message_type == "food":
        return {"next": "food"}

    return {"next": "gym"}
