Feature: complexity memory
  The system remembers the variable, entity and value cast aside 
  during the complexity descent process.

  Rule: Simplification, not oblivion
    Data complexity is not obliviated during the complexity descent process
    but simplified, which means it is just taking new shape. 
  
  Scenario: keep entity, variable and value deleted in memory
    When the expert deletes an entity, variable or value
    Then the system remembers the deleted entity, variable or value
    And the system shows the deleted entity, variable or value to the user 