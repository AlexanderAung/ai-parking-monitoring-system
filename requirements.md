# Car-Parking System with CV

## Problem Statement

Shopping centers need to know which parking spaces are occupied, which vehicles are currently parked, how long they have been parked, and which vehicle corresponds to which parking record. Becasue parking is either a direct revenue stream or major operating cost. 

### Traditional way of solving this manually creates

#### Manual Entry/Exit Bottlenecks:

Traditional methods like manual gate passes, RFID card swiping, or ticket dispensers create traffic congestion during peak hours. These systems can also be costly to maintain due to wear and tear or require specialized hardware that is difficult to scale.

#### High Operational Costs:

Maintaining on-site security personnel solely for manual monitoring is labor-intensive and expensive. Automated, computer-vision-based monitoring offers a more scalable and cost-effective alternative by digitizing the process of identification and record-keeping

## Requirements

### Functional requirements 
1. Detect vehicles' license plates from parking-lot entry camera footage.
2. Record vehicle entry time and save it to the database 
3. Record vehicle exit time and save it to the database
4. Calculate parking duration when the vehicle is detected exiting the gate
5. Store parking records in a database.
6. Display current parking availability.

### Non-functional requirements
1. Detection accuarcy of 90%
