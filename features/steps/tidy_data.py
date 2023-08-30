from behave import given, when, then

class Context:
    def __init__(self):
        self.complexity_memory_feature_worked = False
        self.messy_data = None
        self.tidy_tables = None
        self.icon_displayed = False

context = Context()

# Given steps
@given("complexity memory feature worked")
def step_complexity_memory_feature_worked(context):
    context.complexity_memory_feature_worked = True

# When steps
@when("the user uses the system")
def step_user_uses_the_system(context):
    assert context.complexity_memory_feature_worked, "Complexity memory feature didn't work"
    context.messy_data = []  # Define the messy data here

@then("the system retrieves the messy data")
def step_system_retrieves_messy_data(context, table):
    context.messy_data = table

@then("turns it into a tidy format")
def step_turns_into_tidy_format(context, *tables):
    context.tidy_tables = tables

@then("the user sees an icon showing when the state of the data tidying process")
def step_user_sees_icon_display(context):
    context.icon_displayed = True
