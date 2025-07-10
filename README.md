When booking a vacation, there are several important factors to consider to ensure you have a smooth, enjoyable experience. Here's a checklist to help guide your decision-making: 

While an LLM might help you brainstorm a travel itinerary, an AI agent will go a step further: booking your flights, comparing hotel prices, and scheduling your transportation without needing explicit commands for each step 

1. Destination 

Climate/Weather: Research the weather and seasons to ensure you'll visit during a favorable time. 

Safety: Check travel advisories, crime rates, and general safety for tourists. 

Activities: Consider what activities you want to do (e.g., adventure, sightseeing, relaxation). 

Cultural and Language Differences: Research the local culture, customs, and language to ensure you're prepared. 

2. Budget 

Accommodation: Compare hotel, resort, or vacation rental prices. Check for deals or discounts. 

Flight Costs: Compare airlines, booking times, and flexible dates to find the best prices. 

Dining and Entertainment: Estimate food, entertainment, and transportation costs. 

Travel Insurance: Consider if you need coverage for cancellations, health emergencies, or lost baggage. 

3. Travel Dates 

Seasonality: Consider peak vs. off-peak times; off-peak may save money but could mean fewer activities. 

Duration: How much time can you afford? Plan for the right balance between relaxation and exploration. 

4. Accommodation Type 

Hotel vs. Airbnb vs. Resort: What suits your style? Hotels may offer more amenities; Airbnb offers local experiences; resorts might have all-inclusive deals. 

Location: Choose accommodations that are conveniently located to the attractions you want to visit. 

Reviews: Check reviews for cleanliness, service quality, and proximity to main attractions. 

5. Transportation 

Flights: Compare prices for different airlines, book early for the best deals, and check for layovers or direct flights. 

Local Transportation: Research how to get around at your destination (public transit, rental cars, taxis, or ride-sharing). 

Travel Times: Be aware of travel times between the airport, hotel, and key attractions. 

6. Health and Safety 

Vaccinations and Health Risks: Check if you need any vaccinations or if there are any health risks in the area. 

Travel Insurance: Consider policies that cover medical issues, lost luggage, and cancellations. 

Emergency Contacts: Keep a list of local emergency numbers (e.g., police, hospitals, embassy). 

7. Visa and Documentation 

Passport: Make sure your passport is valid for at least six months after your planned return date. 

Visa Requirements: Verify if you need a visa for the destination country and the application process. 

Travel Authorization: Some countries may require pre-approval or electronic travel authorizations (e.g., ESTA for the U.S.). 

8. Packing and Essentials 

Packing List: Make a list of items you'll need (e.g., clothes, toiletries, tech gadgets, medications). 

Adapters: Ensure you have the correct plug adapters if traveling abroad. 

Local Currency and Payment Options: Check if your destination uses a different currency and inform your bank if you plan to use your credit cards internationally. 

9. Local Culture and Etiquette 

Cultural Norms: Be respectful of local traditions, dress codes, and taboos. 

Language: Learn some basic phrases, especially if you’re traveling to a country with a different language. 

10. Excursions and Tours 

Pre-booking Tours: Research and book any tours or excursions in advance, especially for popular activities or destinations. 

Day Trips: Plan any day trips or excursions you might want to take during your vacation. 

11. Sustainability and Ethics 

Eco-Friendly Options: Consider sustainable travel options (e.g., eco-friendly hotels, responsible tourism). 

Local Impact: Choose businesses and tours that support local communities and cultures. 

12. Communication and Connectivity 

SIM Card or Roaming: Check if you need a local SIM card or if your phone plan covers international roaming. 

Wi-Fi Access: Confirm if your accommodation or public spaces provide Wi-Fi for work or staying in touch. 

13. Backup Plans 

Contingency Plans: Know what to do in case of an emergency or unexpected changes, such as weather disruptions or canceled flights. 

Important Documents: Keep digital copies of important documents like your passport, visa, and reservation details. 

14. Special Needs or Requests 

Accessibility: If you or someone you're traveling with has mobility issues, ensure your accommodation and activities are accessible. 

Dietary Restrictions: Make sure your dietary preferences or restrictions can be accommodated at restaurants or resorts. 

15. Final Check 

Itinerary Details: Double-check flight times, hotel bookings, and any special events or reservations. 

Reminders: Set reminders for booking confirmations, packing, and departure times. 

By keeping these factors in mind, you’ll be better prepared for your vacation and can ensure a stress-free and enjoyable trip! 


i am working for a travel booking portal which helps users with hotels and flight bookings through an online website. I want to build an AI-powered chatbot for travel domain (post booking and pre booking) where the user actions could be classified as (but not limited to):

FlightBookingInquiry (e.g., "Find flights from NYC to LA")
HotelBookingInquiry (e.g., "Book a hotel in London for next week")
CheckFlightStatus (e.g., "Is my flight to Chicago on time?")
ChangeFlight (e.g., "I need to change my flight date")
CancelBooking (e.g., "How do I cancel my reservation?")
BaggagePolicy (e.g., "What's the baggage allowance for United?")
PaymentIssue (e.g., "My payment didn't go through")
RefundRequest (e.g., "I'd like a refund")
GeneralInquiry (e.g., "How does CheapOair work?")
TechnicalSupport (e.g., "The website is not loading")
Complaint (e.g., "I had a bad experience")
Tell me the project design and the flow on how I should build this product. The way i thought about the project is having it divided in 2 steps: 1. NER & Intent Classification 2. RAG based approach (for post booking actions ) However, I would also like to know if using a pre-trained LLM model would be better than developing everything in-house? Do I need to finetune the model on historical chat/call transcription data from the portal. Which method will be easier to build and deploy?

How do i perform prompt engineering using an open -sourced LLM model to perform NER for the above task?

New Booking
Modify Booking
Cancel booking
CCD
Seat Related Queries
Baggage Related Queries
Ancillaries
General Booking Queries


POC Background 

Project Overview 

This document outlines the Proof of Concept (PoC) for implementing a voice-based conversational AI solution in Fareportal's contact center. The objective is to evaluate the feasibility, effectiveness, and potential benefits of integrating an AI-driven voice assistant to enhance customer service operations and identify a long-term partner for Fareportal. We expect to POC multiple vendors through this process to evaluate “fit” for Fareportal  

 

 

Company Background 

Founded in 2002, Fareportal has been creating travel technology for over 20 years that powers leading hybrid travel agencies like CheapOair, OneTravel, Farebuzz and Travelong, a veteran corporate travel agency. Headquartered in New York and has offices in Canada, India, Las Vegas, Mexico, Ukraine, and the United Kingdom. 

Fareportal partners with over 500 airlines, over 1 million hotels, and hundreds of car agencies worldwide. Offering a hybrid business model that combines online booking with personalized trip booking experiences from trained travel agents. Fareportal also provides airfares, hotels, vacation packages, and car rentals 

Proof of Concept (POC) 

POC Objectives 

The Proof-of-Concept Objective is to aid decision-making around platform selection based on the following areas. 

Improve Customer Experience: Provide seamless, 24/7 customer inquiries and issues support. 

Increase Efficiency: Automate routine tasks and inquiries to reduce waiting times and workload on human agents. 

Scalability: Ensure the solution can handle varying volumes of customer interactions effectively. 

Insights and Analytics: Capture and analyze customer interaction data to gain insights for continuous improvement. 

Integration with Fareportal’s internal platforms: Seamless integration with Fareportal’s internal platforms. 

POC Approach 

For the POC approach we are looking to a small number of real business problems with real Fareportal customers. We expect the POC to have multiple phases starting simple (i.e. understanding intent) and then ultimately solving for the customers desired outcome. 

The exact customer journeys we intend to use for the POCs are listed in the Appendix at the end of this document. 

 

POC Scope 

The Scope of the POC would be to automate customer airline ticket booking and support journeys for a limited set of customers and evaluate the performance of the solution based on the mentioned. 

We expect to take a a small number (5%) of real customer calls from the “top” of our existing IVR funnel and use live customers to evaluate the success of Ai-Agents as well as the “partner-fit” 

We expect the POCs to iteratively demonstrate logical progress from simple to complex interactions.  This approach potentially allows vendors to phase the integration timeline to align to these sub-phases – i.e., we understand that more complex interactions will require more integrations with our core platforms. 

Natural Language Understanding (NLU) – Intent Identification 

Speech Recognition and Synthesis - Information Gathering / Customer Qualification 

Handling Multiple Use Cases—Provide Initial Full Services Support, e.g., booking airline tickets based on varied criteria. 

Integration with Existing Systems (IVR, Ticket Management system, etc.) 

 

POC Duration 

We expect the POC will be 3 months to be broken down into multiple phases, including i) Scoping, ii) Integration & Learning, and ii) POC Execution.   We expect that phase ii) and iii) may overlap as complexity increases and availability of integrations. 

Use Cases  

The table below outlines the high-level use case journey /Flow that we expect to perform through the course of the pilot. This list is provided as an initial generic list. We expect this list to be adjusted based on specific solution implementations and will be agreed with individual solutions as part pf the initial POC phases prior to execution. In addition, we may choose to add additional items or remove items from this list during the POC based on lessons learnt during the evaluation. 

  

No. 

Intent 

Keywords 

Action 

1 

New Flight Booking 

Make flight booking, Book a flight 

Gather information about desired flight – To and From, Return or One-way, Departure and Returning Dates, Number of passengers, Cabin Class. Then route to Flight Sales skill along with gathered information 

2 

New Hotel Booking 

Make hotel booking, Book a hotel 

Route to Hotel Sales skill 

3 

New Booking 

Make a booking 

Determine if caller wants to make flight booking. If yes, then follow #1, else directly route to Flight Sales skill 

4 

Existing Booking – Payment Issue 

Card was declined, Payment did not go through, Card verification issue 

Route to CCD or CCV skill 

5 

Existing Booking – Exchange 

Change my flight, Change passenger name, Change billing address 

Route to Exchange skill 

6 

Existing Booking – Cancellation 

Cancel my flight, Cancel my booking 

Route to Cancellation skill 

7 

Existing Booking – Booking confirmation 

What is my booking status, Has my booking been confirmed, Is my booking confirmed, Is my flight confirmed 

Authorize the caller against the booking (API Integration #6) and then provide confirmation to the caller if their booking has been confirmed. After that, ask customer if they need more help, or if they would like to select their seats, and if yes for either, then route to CS skill with the relevant information. If not, gracefully end the call.  

If booking has not been confirmed, tell that to the customer and ask if they need to talk to an agent. If yes, then route to CS skill. 

8 

Existing Booking – Any other query/Unidentified concern 

 

Route to CS skill 

 

For the API integrations, we have mentioned the serial numbers (that are present in a table in section 5.1.2) against each of the step. 

Also based on each of the use cases mentioned below please provide estimated number of Interaction/ Playback are required for Verloop’s learning engine. ( No playback Required as shared by Verloop) 

 

Phase #1 Intent Determination 

Phase Description 

The first phase focuses on correctly understanding the intent of the customer and directing them to the appropriate skill. 

Success Criteria 

Reduction in IVR abandon (%age) and in transfer rate (%age) 


Phase Description 

Voice bot to determine basic understanding of flight details (number of pax, return/one-way, departure and arrival, dates etc.) and handover qualified leads to relevant agents 

Success Criteria 

Phase 1 KPIs + Increase in Agent Conversion Rate (%age), Reduction in New flight booking txn AHT 

 
Phase Description 

VoiceBot to determine customer’s queries and resolve them using booking data. After successful resolution of customer query, pitch add-on products. If the VoiceBot is unable to resolve a customer query, hand it to the relevant agent. 

Success Criteria 

Bot contain rate (%age of calls contained by bot), User authorization rate, Attachment rate (%age of bookings add-ons were attached), CSAT 

