import phonenumbers
from phonenumbers import carrier

# Manual telecom circle mapping (extendable)
TELECOM_CIRCLE_MAP = {
    "85": "Bihar & Jharkhand",
    "91": "Delhi / NCR"
}

def get_telecom_circle(number):
    try:
        phone = phonenumbers.parse(number, "IN")

        if not phonenumbers.is_valid_number(phone):
            return "Invalid Number"

        national_number = str(phone.national_number)
        prefix = national_number[:2]

        telecom_circle = TELECOM_CIRCLE_MAP.get(prefix, "Unknown Circle")
        operator = carrier.name_for_number(phone, "en")

        return {
            "Number": number,
            "Prefix": prefix,
            "Telecom Circle": telecom_circle,
            "Operator": operator
        }

    except phonenumbers.NumberParseException:
        return {"Number": number, "Error": "Invalid format"}

# -------------------------
# Check TWO numbers
# -------------------------
numbers = [
    "+918528833671",  # 85 series
    "+919155864439"   # 91 series
]

for num in numbers:
    result = get_telecom_circle(num)
    print(result)
