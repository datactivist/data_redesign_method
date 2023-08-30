Feature: measure dataset complexity order
  The user wants to know how complex the datasets he intends to use are. 
  The feature will give the user an approximation of such complexity 
  by displaying the complexity order of the data.

  Rule: One dataset, one measure, one reuse
    The user can only measure the complexity of one dataset at a time.
  
  Scenario: measure complexity order of a dataset
    Given the user has a dataset
    When the user asks for the complexity order of the dataset
    Then the user will be given the complexity order of the dataset 
