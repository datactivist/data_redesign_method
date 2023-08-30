from behave import given, when, then

class Context:
    def __init__(self):
        self.expert_action = None
        self.deleted_item = None
        self.system_memory = []
        self.deleted_item_shown = False

context = Context()

# When steps
@when("the expert deletes an entity, variable or value")
def step_expert_deletes_entity_variable_value(context):
    context.expert_action = "delete"

# Then steps
@then("the system remembers the deleted entity, variable or value")
def step_system_remembers_deleted_item(context):
    assert context.expert_action == "delete", "Expert action not set"
    context.deleted_item = "example_deleted_item"  # Replace with actual deleted item
    context.system_memory.append(context.deleted_item)

@then("the system shows the deleted entity, variable or value to the user")
def step_system_shows_deleted_item_to_user(context):
    assert context.deleted_item, "Deleted item not remembered"
    assert context.deleted_item in context.system_memory, "Deleted item not in memory"
    context.deleted_item_shown = True
