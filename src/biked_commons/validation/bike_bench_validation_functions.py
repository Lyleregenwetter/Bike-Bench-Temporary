from typing import List
import torch
import math
from biked_commons.validation.base_validation_function import ValidationFunction

POSITIVE_COLS = ['CS textfield', 'Stack', 'Head angle',
       'Head tube length textfield', 'Seat stay junction0', 'Seat tube length',
       'Seat angle', 'DT Length', 'FORK0R', 'BB diameter', 'ttd', 'dtd', 'csd',
       'ssd', 'Chain stay position on BB', 'SSTopZOFFSET',
       'Head tube upper extension2', 'Seat tube extension2',
       'Head tube lower extension2', 'SEATSTAYbrdgshift', 'CHAINSTAYbrdgshift',
       'SEATSTAYbrdgdia1', 'CHAINSTAYbrdgdia1', 'Dropout spacing',
       'Wall thickness Bottom Bracket', 'Wall thickness Top tube',
       'Wall thickness Head tube', 'Wall thickness Down tube',
       'Wall thickness Chain stay', 'Wall thickness Seat stay',
       'Wall thickness Seat tube', 'Wheel diameter front', 'RDBSD',
       'Wheel diameter rear', 'FDBSD', 'BB length',
       'Head tube diameter', 'Wheel cut', 'Seat tube diameter', 'Number of cogs',
       'Number of chainrings', 'FIRST color R_RGB',
       'FIRST color G_RGB', 'FIRST color B_RGB', 'SPOKES composite front',
       'SPOKES composite rear', 'SBLADEW front', 'SBLADEW rear',
       'Saddle length', 'Saddle height', 'Down tube diameter',
       'Seatpost LENGTH']

class SaddleHeightTooSmall(ValidationFunction):
    def friendly_name(self) -> str:
        return "Saddle height too small"

    def variable_names(self) -> List[str]:
        return ["Saddle height", "Seat tube length"]

    def validate(self, designs: torch.tensor) -> torch.tensor:
        saddle_height, seat_tube_length = designs[:, :len(self.variable_names())].T
        return (seat_tube_length + 40) - saddle_height


class SeatPostTooShort(ValidationFunction):
    def friendly_name(self) -> str:
        return "Seat post too short"

    def variable_names(self) -> List[str]:
        return ["Seat tube length", "Seatpost LENGTH", "Saddle height"]

    def validate(self, designs: torch.Tensor) -> torch.Tensor:
        seat_tube_length, seatpost_length, saddle_height = designs[:, :len(self.variable_names())].T
        return saddle_height - (seat_tube_length + seatpost_length + 30) 


# class BsdRearTooSmall(ValidationFunction):
#     def friendly_name(self) -> str:
#         return "Bsd rear too small"

#     def variable_names(self) -> List[str]:
#         return ["RDBSD", "RDERD"]

#     def validate(self, designs: torch.tensor) -> torch.tensor:
#         RDBSD, RDERD = designs[:, :len(self.variable_names())].T
#         return RDBSD - RDERD


# class BsdFrontTooSmall(ValidationFunction):
#     def friendly_name(self) -> str:
#         return "Bsd front too small"

#     def variable_names(self) -> List[str]:
#         return ["FDBSD", "FDERD"]

#     def validate(self, designs: torch.tensor) -> torch.tensor:
#         FDBSD, FDERD = designs[:, :len(self.variable_names())].T
#         return FDBSD - FDERD


class HeadTubeLowerExtensionTooGreat(ValidationFunction):
    def friendly_name(self) -> str:
        return "Head tube lower extension too great"

    def variable_names(self) -> List[str]:
        return ["Head tube length textfield", "Head tube lower extension2"]

    def validate(self, designs: torch.tensor) -> torch.tensor:
        head_tube_length, head_tube_lower_extension = designs[:, :len(self.variable_names())].T
        return head_tube_lower_extension - head_tube_length


class HeadTubeLengthTooGreat(ValidationFunction):
    def friendly_name(self) -> str:
        return "Head tube length too great"

    def variable_names(self) -> List[str]:
        return ["Head tube length textfield", "Head tube upper extension2", "Head tube lower extension2"]

    def validate(self, designs: torch.tensor) -> torch.tensor:
        head_tube_length, head_tube_upper_extension, head_tube_lower_extension = designs[:,
                                                                                 :len(self.variable_names())].T
        return (head_tube_upper_extension + head_tube_lower_extension) - head_tube_length


class PositiveValueNegative(ValidationFunction):
    def friendly_name(self) -> str:
        return "Certain parameters must be positive"

    def variable_names(self) -> List[str]:
        return POSITIVE_COLS

    def validate(self, designs: torch.tensor) -> torch.tensor:
        all_clipped = torch.clip(-designs, min=0)    
        sum = torch.sum(all_clipped, dim=1)
        return sum


class ChainStayShouldSmallerThanWheelRadius(ValidationFunction):
    def friendly_name(self) -> str:
        return "Chain stay should be greater than wheel radius"

    def variable_names(self) -> List[str]:
        return ["CS textfield", "Wheel diameter rear"]

    def validate(self, designs: torch.tensor) -> torch.tensor:
        CS_textfield, RD = designs[:, :len(self.variable_names())].T
        return (RD/2) - CS_textfield
    
class ChainStaySmallerThanBB(ValidationFunction):
    def friendly_name(self) -> str:
        return "Chain stay should be greater than BB"

    def variable_names(self) -> List[str]:
        return ["CS textfield", "BB textfield"]

    def validate(self, designs: torch.tensor) -> torch.tensor:
        CS_textfield, BB_textfield = designs[:, :len(self.variable_names())].T
        return BB_textfield - CS_textfield

class SeatStaySmallerThanWheelRadius(ValidationFunction):
    def friendly_name(self) -> str:
        return "Seat stay should be greater than wheel radius"

    def variable_names(self) -> List[str]:
        return ["CS textfield", "BB textfield","Seat tube length", "Seat stay junction0", "Seat angle", "Wheel diameter rear"]

    def validate(self, designs: torch.tensor, eps = 1e-6) -> torch.tensor:
        CS_textfield, BB_textfield, Seat_tube_length, Seat_stay_junction0, Seat_angle, RD = designs[:, :len(self.variable_names())].T
        Seat_angle_rad = (Seat_angle * math.pi) / 180
        x = Seat_tube_length-(BB_textfield/torch.sin(Seat_angle_rad))-Seat_stay_junction0
        y = BB_textfield/torch.tan(Seat_angle_rad)
        z = torch.sqrt(torch.clip((CS_textfield ** 2)-(BB_textfield ** 2), min=eps))
        h = z-y
        g = torch.sqrt(h**2 + x**2 - 2*h*x*torch.cos(Seat_angle_rad))
        return (RD / 2) - g
    
class DownTubeCantReachHeadTube(ValidationFunction):
    def friendly_name(self) -> str:
        return "Down tube must reach head tube"

    def variable_names(self) -> List[str]:
        return ["Stack", "Head tube length textfield", "Head tube lower extension2", "Head angle", "DT Length"]

    def validate(self, designs: torch.tensor) -> torch.tensor:
        Stack, Head_tube_length_textfield, Head_tube_lower_extension2, Head_angle, DT_length = designs[:, :len(self.variable_names())].T
        # Extract variables from the DataFrame
        HTL = Head_tube_length_textfield
        HTLX = Head_tube_lower_extension2
        HTA = (Head_angle * math.pi) / 180  # Convert degrees to radians
        DTL = DT_length

        # Calculate DTJY and DTJX
        DTJY = Stack - (HTL - HTLX) * torch.sin(HTA)

        return DTJY - DTL

class PedalIntersectsFrontWheel(ValidationFunction):
    def friendly_name(self) -> str:
        return "The pedal shouldn't intersect the front wheel"

    def variable_names(self) -> List[str]:
        return ["Stack", "Head tube length textfield", "Head tube lower extension2", "Head angle", "BB textfield", "DT Length", "FORK0R", "Wheel diameter rear", "Wheel diameter front"]

    def validate(self, designs: torch.tensor, eps=1e-6) -> torch.tensor:
        Stack, Head_tube_length_textfield, Head_tube_lower_extension2, Head_angle, BB_textfield, DT_length, fork0r, WDR, WDF = designs[:, :len(self.variable_names())].T
        # Extract variables from the DataFrame
        HTL = Head_tube_length_textfield
        HTLX = Head_tube_lower_extension2
        HTA = (Head_angle * math.pi) / 180  # Convert degrees to radians
        BBD = BB_textfield
        FTY = BBD - WDR / 2 + WDF / 2
        DTL = DT_length

        # Calculate DTJY and DTJX
        DTJY = Stack - (HTL - HTLX) * torch.sin(HTA)

        DTJX = torch.sqrt(torch.clip(DTL ** 2 - DTJY ** 2, min=eps))

        # Calculate FWX and FCD
        FWX = DTJX + (DTJY - BBD) / torch.tan(HTA)
        shift = fork0r/torch.sin(HTA)
        FWX = FWX + shift

        FCD = torch.sqrt(FWX ** 2 + BBD ** 2)
        wheel_radius = WDF/2
        crank_length = 172.5
        return  (crank_length + wheel_radius + 40) - FCD
    
class CrankHitsGroundInLowestPosition(ValidationFunction):
    def friendly_name(self) -> str:
        return "The crank shouldn't hit the ground when it is in its lower position"

    def variable_names(self) -> List[str]:
        return ["BB textfield", "Wheel diameter rear"]

    def validate(self, designs: torch.tensor) -> torch.tensor:
        BB_textfield, WDR = designs[:, :len(self.variable_names())].T
        # Extract variables from the DataFrame
        wheel_radius = WDR / 2
        crank_length = 172.5
        return  (crank_length + BB_textfield) - wheel_radius

class RGBvalueGreaterThan255(ValidationFunction): #less than 0 covered in PositiveValueNegative
    def friendly_name(self) -> str:
        return "RGB value should be less than 255"

    def variable_names(self) -> List[str]:
        return ["FIRST color R_RGB", "FIRST color G_RGB", "FIRST color B_RGB"]

    def validate(self, designs: torch.tensor) -> torch.tensor:
        color_overflow = designs - 255
        overflow_clipped = torch.clip(color_overflow, min=0)
        total = torch.sum(overflow_clipped, dim=1)
        #return total if total > 0 else sum of color_overflow (required for calculation of default weights)
        mask = total > 0
        return total * mask.float() + color_overflow.sum(dim=1) * (1 - mask.float())




bike_bench_validation_functions: List[ValidationFunction] = [
    SaddleHeightTooSmall(),
    SeatPostTooShort(),
    HeadTubeLowerExtensionTooGreat(),
    HeadTubeLengthTooGreat(),
    PositiveValueNegative(),
    ChainStayShouldSmallerThanWheelRadius(),
    ChainStaySmallerThanBB(),
    SeatStaySmallerThanWheelRadius(),
    DownTubeCantReachHeadTube(),
    PedalIntersectsFrontWheel(),
    CrankHitsGroundInLowestPosition(),
    RGBvalueGreaterThan255()
]