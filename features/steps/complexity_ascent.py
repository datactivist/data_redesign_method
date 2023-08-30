from behave import given, when, then

class Context:
    def __init__(self):
        self.memory_worked = False
        self.tidy_data_worked = False
        self.passed_one_day = False
        self.user_reached_lowest_complexity = False
        self.user_asks_for_ascent = False
        self.appointment_scheduled = False
        self.copilot_role_set = False
        self.user_agrees = False
        self.tools_shown = False
        self.tool_used = False
        self.new_complexity_evaluated = False
        self.reuses_shown = False

context = Context()

# Given steps
@given("feature")
def step_feature(context):
    pass

@given("memory worked")
def step_memory_worked(context):
    context.memory_worked = True

@given("tidy data worked")
def step_tidy_data_worked(context):
    context.tidy_data_worked = True

@given("one day at least passed since the last complexity descent")
def step_passed_one_day(context):
    context.passed_one_day = True

@given("user reached the lowest complexity order")
def step_user_reached_lowest_complexity(context):
    context.user_reached_lowest_complexity = True

# When steps
@when("the user asks for complexity ascent")
def step_user_asks_for_ascent(context):
    context.user_asks_for_ascent = True

@when("users agrees")
def step_users_agrees(context):
    context.user_agrees = True

@when("the user is done using one tool")
def step_user_done_using_tool(context):
    context.tool_used = True

# Then steps
@then("he gets an appointment with the data expert")
def step_get_appointment_with_expert(context):
    assert context.memory_worked, "Memory feature didn't work"
    assert context.tidy_data_worked, "Tidy data feature didn't work"
    assert context.passed_one_day, "Not enough time has passed since last descent"
    assert context.user_reached_lowest_complexity, "User hasn't reached lowest complexity"
    assert context.user_asks_for_ascent, "User didn't ask for ascent"
    context.appointment_scheduled = True

@then("the expert he will play the role of a copilot")
def step_expert_plays_copilot_role(context):
    assert context.appointment_scheduled, "Appointment not scheduled"
    context.copilot_role_set = True

@then("he is showed complexity ascent tools")
def step_show_ascent_tools(context):
    assert context.copilot_role_set, "Copilot role not set"
    assert context.user_agrees, "User didn't agree"
    context.tools_shown = True

@then("the expert/copilot evaluate the new complexity order")
def step_evaluate_new_complexity(context):
    assert context.tools_shown, "Tools not shown"
    assert context.tool_used, "User didn't use tool"
    context.new_complexity_evaluated = True

@then("he shows the user reuses option now available")
def step_show_available_reuses(context):
    assert context.new_complexity_evaluated, "Complexity not evaluated"
    context.reuses_shown = True
