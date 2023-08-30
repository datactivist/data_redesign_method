Feature: tidy data
  Feature that turns messy data cast aside during
  the complexity descent process into tidy data.

  Scenario: tidy data cast aside
    Given complexity memory feature worked
    When the users uses the system
    Then the system retrieves the messy data

    | Entity       | Attribute | Value        | Order ID | Product Name | Quantity | Total Amount | Method   | Address     | Method      | Amount     |
    | Customer     | Name      | John         |          |              |          |              |          |             |             |            |
    |              | Phone     | 555-123-4567 |          |              |          |              |          |             |             |            |
    |              | Address   | 123 Main St  |          |              |          |              |          |             |             |            |
    | Order        |           |              | 987654   |              |          |              |          |             |             |            |
    |              |           |              |          | Widget       |          |              |          |             |             |            |
    |              |           |              |          |              | 3        |              |          |             |             |            |
    |              |           |              |          |              |          | $150.00      |          |             |             |            |
    | Shipping     |           |              |          |              |          |              | Express  |             |             |            |
    |              |           |              |          |              |          |              |          | 456 Elm St |             |            |
    | Payment      |           |              |          |              |          |              |          |             | Credit Card |            |
    |              |           |              |          |              |          |              |          |             |             | $150.00    |

    
    Then turns it into a tidy format 

    |   Entity   |     Attribute     |      Value      |
    |  Customer  |       Name        |      John       |
    |  Customer  |   Phone Number    |   555-123-4567  |
    |  Customer  |   Address         |   123 Main St   |


    |  Entity  |   Attribute    |   Value    |
    |  Order   |   Order ID     |   987654   |
    |  Order   |  Product Name  |   Widget   |
    |  Order   |   Quantity     |     3      |
    |  Order   |  Total Amount  |  $150.00   |


    |  Entity   |  Attribute  |    Value    |
    |  Shipping |   Method    |   Express   |
    |  Shipping |   Address   |  456 Elm St |


    |  Entity  |  Attribute   |     Value    |
    |  Payment |   Method     |  Credit Card |
    |  Payment |   Amount     |     $150.00  |


    And the user sees an icon showing when the state of the data tidying process