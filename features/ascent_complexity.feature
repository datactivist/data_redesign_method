Feature: complexity ascent
  Feature that reaugment the complexity order of the data

  Rule: Reuse type as ultimate target
    One complexity order relates to one reuse type.
    When the complexity order reached by the user meets reuse type targeted by the user
    then the data redesign method has fulfilled its mission
  
  Rule: Graphic interactions
    Complexity ascent can only happen through graphic interactions
    with clear and distinct entites, attributes and entities.

  Scenario: complexity ascent
    Given feature
    * memory worked
    * tidy data worked
    * one day at least passed since the last complexity descent 
    * user reached the lowest complexity order
    When the user asks for complexity ascent
    Then he gets an appointment with the data expert
    Then the expert he will play the role of a copilot
    When users agrees
    Then he is showed complexity ascent tolls
    When the user is done using one tool 
    Then the expert/copilot evaluate the new complexity order 
    Then he shows the user reuses option now available 