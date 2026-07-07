from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from typing_extensions import TypedDict

from helpers import classify_message, router
from agents import food_agent, gym_agent


class MessageClassifier(BaseModel):
    message_type: Literal["food", "gym"]


class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None



# Build the state graph
graph_builder = StateGraph(State)

# Add nodes to the graph
graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("food", food_agent)
graph_builder.add_node("gym", gym_agent)

# Add edges to the graph
graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")

# Add conditional edges based on the router's output
graph_builder.add_conditional_edges(
    "router",
    lambda state: state.get("next"),
    {"food": "food", "gym": "gym"}
)

# Add edges from the agents to the END node
graph_builder.add_edge("food", END)
graph_builder.add_edge("gym", END)

# Compile the graph
graph = graph_builder.compile()


def main():
    state = {"messages": [], "message_type": None}

    while True:
        user_input = input("Message: ")
        if user_input == "exit" or user_input == "quit" or user_input == "q":
            print("Good bye")
            break

        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": user_input}
        ]

        state = graph.invoke(state)

        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            print(f"Assistant: {last_message.content}")


if __name__ == "__main__":
    main()
