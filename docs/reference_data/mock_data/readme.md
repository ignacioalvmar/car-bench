# Mock Data Generation

## Current Mock Data for the Benchmark
- Navigation
    - locations.jsonl: a database of 48 real locations within Europe including their name, an id, and position (longitude, latitude).
    - pois.jsonl: a database of 130693 synthetic POIs including their name, an id, position (longitude, latitude), category, opening hours, phone number, and corresponding location id or corresponding route id. There are 3 POIs of each category around each location, and 1 POI of each category on every 50km along every route.
    - Routes (in total 1763869 routes)
        - routes_location_location.jsonl: a database of 1000 synthetic routes between locations including their route id, start id, destination id, name via, distance in km, duration in hours and minutes, road types, if it includes toll roads, and aliases (first/second/thirs, fastest, shortest). Every location <--> location pair has 3 route alternatives.
        - routes_location_poi.jsonl: same as location <--> location routes, but location <--> poi routes.
        - routes_poi_location.jsonl: same as location <--> location routes, but poi <--> location routes.
        - routes_index.jsonl: database of each route as first key for faster lookup.
        - routes_metadata.jsonl: metadata needed for poi <--> poi routes: as poi <--> poi routes have high cardinality, we only store basic metadata and generate the actual route deterministically during runtime from the metadata.
    - weather.jsonl: a database of 48 synthetic weather data for each location for one day.
- Productivity and Communication
    - contacts.jsonl: a database of 100 synthetic contacts including their first name, last name, email, phone number, and id.
    - calendars.jsonl: a database of 100 synthetic calendars including two appointements per day. Each calendar has an id and a date (year, month, day), each appointement has a start time, an end time, a duration, a location, an attendees list (aligning with the contacts database), and a title.

Check `../tools` to see which tools retrieve the current mock data.


### Experience of Mock Data Generation

In general, it involves the following stages:

1. Design the type and schema of each database. Can use LLM for co-brainstorming but has to be human decided as it is the foundation of everything else.
2. For each schema, figure out which parts can be programmaticly generated and which parts need LLM. For example,
    - POI names and user names (Sara, John, Noah) need LLM generation
    - Route distances and durations can be generated via code
3. Use LLM to generate seed data (first names, last names, etc.), then use a program to compose them with other code generated data.






