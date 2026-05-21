class ConversationState:
    def __init__(self):
        self.pending_action = None
        self.last_prompt_type = None
        self.data = {}

    def set_pending(self, action: str, data: dict | None = None) -> None:
        self.pending_action = action
        self.data = data or {}

    def clear(self) -> None:
        self.pending_action = None
        self.data = {}


conversation_state = ConversationState()
STATE = conversation_state
