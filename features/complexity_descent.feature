Feature: get down from upper to lower complexity order
  The user wants to reuse a dataset to create a new indicator. 
  The user found out that the complexity order of the datasets is 3. 
  He wants to lower the complexity order to 2, so he gets in touch with a data expert. 
  He meets the expert and at the end of the process, the complexity order of the dataset is 2.

  Rule: One user, one expert
    One user can only be in touch with one expert throughout the whole process.
  
  Scenario: complexity descent
    Given the user used the measure complexity feature
    When the user wants to lower the complexity order of the dataset
    Then he gets in touch with a data expert
    When the appointment starts
    Then the expert asks the user what indicator he wants to create
    When the user answers
    Then the expert asks the user what dataset he wants to use
    When the user answers
    Then the expert asks the user what complexity order he wants to reach
    When the user answers
    Then the expert begins the process of lowering the complexity order
    When the process is over
    Then the complexity order of the dataset is lowered
    Then he explains what the user can do with such complexity order 
    Then the appointment ends

