from main import State
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline


pipe = pipeline(
    "text-generation",
    model="gpt2",
    max_new_tokens=100,
)

llm = HuggingFacePipeline(pipeline=pipe)


def food_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
        {"role": "system",
         "content": """You are a food advicer. Focus on the health aspects based on facts and studies.
                        You may ask thoughtful questions to help explore the dietary habits more deeply."""
         },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}


def gym_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
        {"role": "system",
         "content": """You are a gym assistant.
            Provide clear, concise answers based on facts and studies.
            Do not address anything about food in your responses."""
         },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}
