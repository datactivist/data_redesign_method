from behave import given, when, then

class Context:
    def __init__(self):
        self.feature_used = False
        self.lower_complexity = False
        self.appointment_started = False
        self.user_answered_indicator = False
        self.user_answered_dataset = False
        self.user_answered_order = False
        self.process_started = False
        self.process_completed = False
        self.explanation_given = False

context = Context()

# Given steps
@given("the user used the measure complexity feature")
def step_user_used_measure_complexity_feature(context):
    context.feature_used = True

# When steps
@when("the user wants to lower the complexity order of the dataset")
def step_user_wants_to_lower_complexity_order(context):
    context.lower_complexity = True

@when("the appointment starts")
def step_appointment_starts(context):
    context.appointment_started = True

@when("the user answers")
def step_user_answers(context):
    if "what indicator" in context.active_step.text:
        context.user_answered_indicator = True
    elif "what dataset" in context.active_step.text:
        context.user_answered_dataset = True
    elif "what complexity order" in context.active_step.text:
        context.user_answered_order = True

@when("the expert begins the process of lowering the complexity order")
def step_expert_begins_lowering_complexity(context):
    assert context.appointment_started, "Appointment hasn't started"
    assert context.user_answered_indicator, "User didn't answer indicator question"
    assert context.user_answered_dataset, "User didn't answer dataset question"
    assert context.user_answered_order, "User didn't answer complexity order question"
    context.process_started = True

@when("the process is over")
def step_process_is_over(context):
    assert context.process_started, "Process hasn't started"
    context.process_completed = True

@when("he explains what the user can do with such complexity order")
def step_explain_user_capabilities(context):
    assert context.process_completed, "Process isn't completed"
    context.explanation_given = True

# Then steps
@then("he gets in touch with a data expert")
def step_user_gets_in_touch_with_data_expert(context):
    assert context.feature_used, "User didn't use the feature"

@then("the expert asks the user what indicator he wants to create")
def step_expert_asks_user_about_indicator(context):
    assert context.appointment_started, "Appointment hasn't started"

@then("the expert asks the user what dataset he wants to use")
def step_expert_asks_user_about_dataset(context):
    assert context.user_answered_indicator, "User didn't answer indicator question"

@then("the expert asks the user what complexity order he wants to reach")
def step_expert_asks_user_about_complexity_order(context):
    assert context.user_answered_dataset, "User didn't answer dataset question"

@then("the complexity order of the dataset is lowered")
def step_complexity_order_lowered(context):
    assert context.process_completed, "Process isn't completed"

@then("he explains what the user can do with such complexity order")
def step_explain_user_capabilities(context):
    assert context.explanation_given, "Explanation not given"

