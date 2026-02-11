import datetime
import json
import random

from car_bench.envs.car_voice_assistant.mock_data import car_va_data_manager

# Sample data
FIRST_NAMES = [
    "Alice",
    "Bob",
    "Charlie",
    "David",
    "Eva",
    "Frank",
    "Grace",
    "Helen",
    "Isaac",
    "Jack",
    "Karen",
    "Leo",
    "Mia",
    "Nathan",
    "Olivia",
    "Paul",
    "Quinn",
    "Rachel",
    "Samuel",
    "Tina",
    "Umar",
    "Vera",
    "William",
    "Xavier",
    "Yasmine",
    "Zane",
    "Sophia",
    "Liam",
    "Noah",
    "Emma",
]
LAST_NAMES = [
    "Smith",
    "Johnson",
    "Brown",
    "Taylor",
    "Anderson",
    "Thomas",
    "White",
    "Harris",
    "Martin",
    "Clark",
    "Lewis",
    "Walker",
    "Hall",
    "Allen",
    "Young",
    "King",
    "Wright",
    "Scott",
    "Green",
    "Baker",
    "Adams",
    "Nelson",
    "Carter",
    "Mitchell",
    "Perez",
    "Roberts",
    "Evans",
    "Turner",
    "Phillips",
    "Campbell",
]
LOCATIONS = [
    "Conference Room A",
    "Conference Room B",
    "Zoom",
    "Google Meet",
    "Microsoft Teams",
    "Main Auditorium",
    "Boardroom",
    "Training Room 1",
    "Training Room 2",
    "Executive Lounge",
    "Cafeteria",
    "Outdoor Patio",
    "Virtual Meeting Room",
    "Town Hall",
    "Project War Room",
    "Breakout Room 1",
    "Breakout Room 2",
    "Future Room",
    "HR Meeting Room",
    "IT Conference Room",
    "Engineering Hub",
    "Marketing Suite",
    "Sales War Room",
    "Finance Department",
    "Customer Support Office",
    "Rooftop Lounge",
    "Library",
    "Innovation Lab",
    "Design Studio",
    "Strategy Room",
]
TOPICS = [
    "Project Update",
    "Budget Discussion",
    "Team Sync",
    "Client Call",
    "Strategy Planning",
    "Product Launch",
    "Quarterly Review",
    "Sales Strategy",
    "Marketing Campaign",
    "Staff Meeting",
    "Performance Review",
    "Risk Management",
    "Innovation Brainstorm",
    "Partnership Discussion",
    "Customer Feedback",
    "Team Building",
    "Leadership Development",
    "Company Vision",
    "Financial Planning",
    "Legal Compliance",
    "Technology Roadmap",
    "Brand Positioning",
    "Customer Retention",
    "Employee Wellness",
    "Sustainability Initiatives",
    "Product Development",
    "Market Research",
    "Supply Chain Optimization",
    "Employee Engagement",
    "Diversity and Inclusion",
]
EMAIL_ENDINGS = [
    "@gmail.com",
    "@yahoo.com",
    "@outlook.com",
    "@hotmail.com",
    "@icloud.com",
    "@aol.com",
    "@protonmail.com",
    "@zoho.com",
    "@andex.com",
]


# Generate calendars.jsonl
def generate_meetings(date_obj):
    """Generate meetings for a specific date"""
    start_times_first_meeting = [
        datetime.datetime(
            date_obj["year"], date_obj["month"], date_obj["day"], hour, 30
        )
        for hour in range(12, 16)
    ] + [
        datetime.datetime(date_obj["year"], date_obj["month"], date_obj["day"], hour, 0)
        for hour in range(13, 15)
    ]
    start_times_second_meeting = [
        datetime.datetime(
            date_obj["year"], date_obj["month"], date_obj["day"], hour, 30
        )
        for hour in range(14, 18)
    ] + [
        datetime.datetime(date_obj["year"], date_obj["month"], date_obj["day"], hour, 0)
        for hour in range(14, 19)
    ]
    random.shuffle(start_times_first_meeting)
    random.shuffle(start_times_second_meeting)
    first_meeting = random.choice(start_times_first_meeting)
    duration = random.choice(["30min", "60min"])
    second_meeting_start_options = [
        t
        for t in start_times_second_meeting
        if t >= first_meeting + datetime.timedelta(hours=1)
    ]
    if not second_meeting_start_options:
        return []  # Fail-safe
    second_meeting = random.choice(second_meeting_start_options)
    second_duration = random.choice([30, 60])
    return [
        {
            "start": {
                "hour": first_meeting.strftime("%H"),
                "minute": first_meeting.strftime("%M"),
            },
            "duration": duration,
        },
        {
            "start": {
                "hour": second_meeting.strftime("%H"),
                "minute": second_meeting.strftime("%M"),
            },
            "duration": second_duration,
        },
    ]


def generate_random_date():
    """Generate a random date in 2025"""
    # Generate dates within 2025 (avoiding weekends for business meetings)
    year = 2025
    month = random.randint(1, 12)

    # Get valid days for the month
    if month in [1, 3, 5, 7, 8, 10, 12]:
        max_day = 31
    elif month in [4, 6, 9, 11]:
        max_day = 30
    else:  # February
        max_day = 28  # 2025 is not a leap year

    day = random.randint(1, max_day)

    # Check if it's a weekend (Monday=0, Sunday=6)
    date_obj = datetime.date(year, month, day)
    if date_obj.weekday() >= 5:  # Saturday or Sunday
        # Try again (simple approach)
        return generate_random_date()

    return {"year": year, "month": month, "day": day}


def main():
    import os

    file_path = os.path.abspath(os.path.dirname(__file__))
    if os.path.exists(f"{file_path}/calendars.jsonl"):
        os.remove(f"{file_path}/calendars.jsonl")
    with open(f"{file_path}/calendars.jsonl", "w") as f:
        pass
    if os.path.exists(f"{file_path}/contacts.jsonl"):
        os.remove(f"{file_path}/contacts.jsonl")
    with open(f"{file_path}/contacts.jsonl", "w") as f:
        pass

    # Generate contacts.jsonl
    contacts = []
    used_names = set()  # Track already used name combinations

    with open(f"{file_path}/contacts.jsonl", "w") as f:
        for _ in range(100):
            # Generate a unique name combination
            while True:
                first_name = random.choice(FIRST_NAMES)
                last_name = random.choice(LAST_NAMES)
                name_key = f"{first_name}_{last_name}"

                if name_key not in used_names:
                    used_names.add(name_key)
                    break

            contact_id = f"con_{random.randint(1000, 9999)}"
            phone_number = (
                f"+49 {random.randint(100, 999)} {random.randint(100000, 999999)}"
            )
            email = f"{first_name.lower()}.{last_name.lower()}{random.randint(1000,9999)}{random.choice(EMAIL_ENDINGS)}"

            contact = {
                "id": contact_id,
                "name": {"first_name": first_name, "last_name": last_name},
                "phone_number": phone_number,
                "email": email,
            }
            contacts.append(contact)
            f.write(json.dumps(contact) + "\n")

    location_data = car_va_data_manager.locations
    location_data_names = [location["name"] for location in location_data.values()]

    with open(f"{file_path}/calendars.jsonl", "w") as f:
        for _ in range(100):  # Generate 30 days of data
            date_obj = generate_random_date()
            meetings = generate_meetings(date_obj)
            for meeting in meetings:
                meeting["location"] = random.choice(location_data_names)
                meeting["attendees"] = random.sample(
                    [c["id"] for c in contacts], k=random.randint(2, 5)
                )
                meeting["topic"] = random.choice(TOPICS)
            calendar_entry = {
                "id": f"cal_{random.randint(1000, 9999)}",
                "date": date_obj,
                "meetings": meetings,
            }
            f.write(json.dumps(calendar_entry) + "\n")

    print("JSON files generated successfully.")


if __name__ == "__main__":
    main()
