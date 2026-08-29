from langgraph.checkpoint.memory import InMemorySaver

def get_checkpointer():
    return InMemorySaver()