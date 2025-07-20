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


class HeadTubeLowerExtensionLongerThanHeadTube(ValidationFunction):
    def friendly_name(self) -> str:
        return "Head tube lower extension longer than head tube"

    def variable_names(self) -> List[str]:
        return ["Head tube length textfield", "Head tube lower extension2"]

    def validate(self, designs: torch.tensor) -> torch.tensor:
        head_tube_length, head_tube_lower_extension = designs[:, :len(self.variable_names())].T
        return head_tube_lower_extension - head_tube_length


class HeadTubeUpperExtensionAndLowerExtensionOverlap(ValidationFunction):
    def friendly_name(self) -> str:
        return "Head tube upper extension and lower extension overlap"

    def variable_names(self) -> List[str]:
        return ["Head tube length textfield", "Head tube upper extension2", "Head tube lower extension2"]

    def validate(self, designs: torch.tensor) -> torch.tensor:
        head_tube_length, head_tube_upper_extension, head_tube_lower_extension = designs[:,
                                                                                 :len(self.variable_names())].T
        return (head_tube_upper_extension + head_tube_lower_extension) - head_tube_length


class StrictlyPositiveParameterIsNegative(ValidationFunction):
    def friendly_name(self) -> str:
        return "Strictly postive parameter is negative"

    def variable_names(self) -> List[str]:
        return POSITIVE_COLS

    def validate(self, designs: torch.tensor) -> torch.tensor:
        all_clipped = torch.clip(-designs, min=0)    
        return torch.sum(all_clipped, dim=1)


class ChainStaySmallerThanRearWheelRadius(ValidationFunction):
    def friendly_name(self) -> str:
        return "Chain stay smaller than rear wheel radius"

    def variable_names(self) -> List[str]:
        return ["CS textfield", "Wheel diameter rear"]

    def validate(self, designs: torch.tensor) -> torch.tensor:
        CS_textfield, RD = designs[:, :len(self.variable_names())].T
        return (RD/2) - CS_textfield
    
class ChainStayShorterThanBBDrop(ValidationFunction):
    def friendly_name(self) -> str:
        return "Chain stay shorter than BB drop"

    def variable_names(self) -> List[str]:
        return ["CS textfield", "BB textfield"]

    def validate(self, designs: torch.tensor) -> torch.tensor:
        CS_textfield, BB_textfield = designs[:, :len(self.variable_names())].T
        return BB_textfield - CS_textfield

class SeatStaySmallerThanRearWheelRadius(ValidationFunction):
    def friendly_name(self) -> str:
        return "Seat stay smaller than rear wheel radius"

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

class SeatTubeIntersectsRearWheel(ValidationFunction):
    def friendly_name(self) -> str:
        return "Seat Tube Intersects Rear Wheel2"

    def variable_names(self) -> List[str]:
        return ["CS textfield", "BB textfield","Seat tube length", "Seat stay junction0", "Seat angle","Seat tube diameter", "Seat tube type OHCLASS: 0", "Wheel cut", "Wheel diameter rear"]

    def validate(self, designs: torch.tensor, eps = 1e-6) -> torch.tensor:
        CS_textfield, BB_textfield, Seat_tube_length, Seat_stay_junction0, Seat_angle, Seat_tube_diameter, Seat_tube_type, Wheel_cut, RD = designs[:, :len(self.variable_names())].T
        Seat_angle_rad = (Seat_angle * math.pi) / 180
        x = Seat_tube_length-(BB_textfield/torch.sin(Seat_angle_rad))-Seat_stay_junction0 # length along seat tube from stay function to horiz interc of wheel
        y = BB_textfield/torch.tan(Seat_angle_rad)                                      #y distance from the above intercept point to bb
        z = torch.sqrt(torch.clip((CS_textfield ** 2)-(BB_textfield ** 2), min=eps))    #Horiz distance from axle to BB
        h = z-y                                                                         #Horiz distance from axle to st intercept
        # g = torch.sqrt(h**2 + x**2 - 2*h*x*torch.cos(Seat_angle_rad))                   #SS Length
        j = (h*torch.sin(Seat_angle_rad))
        
        mask = (Seat_tube_type == 1)                                                    #If aero tube, use logic based on wheel cut
        #q_true = j - (Seat_tube_diameter / 2) + ((Wheel_cut-RD) / 2)
        q_true = torch.where(Wheel_cut < RD, j - 40.9, (RD / 2) + ((Wheel_cut - RD) / 2))
        q_false = j - (Seat_tube_diameter/2) 
        q = torch.where(mask, q_true, q_false)
        
        return (RD / 2) - q

class DownTubeCantReachHeadTube(ValidationFunction):
    def friendly_name(self) -> str:
        return "Down tube can't reach head tube"

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
    
class RearWheelCutoutSeversSeatTube(ValidationFunction):
    def friendly_name(self) -> str:
        return "Rear wheel cutout severs seat tube"

    def variable_names(self) -> List[str]:
        return ["CS textfield", "BB textfield","Seat tube length", "Seat stay junction0", "Seat angle", "Seat tube type OHCLASS: 0", "Wheel cut"] # replace Seat tube type  (mixed df) with Seat tube type OHCLASS: 0 (dataset)

    def validate(self, designs: torch.tensor, eps = 1e-6) -> torch.tensor: #same logic as: SeatTubeIntersectsRearWheel
        CS_textfield, BB_textfield, Seat_tube_length, Seat_stay_junction0, Seat_angle, Seat_tube_type, Wheel_cut = designs[:, :len(self.variable_names())].T
        Seat_angle_rad = (Seat_angle * math.pi) / 180
        x = Seat_tube_length-(BB_textfield/torch.sin(Seat_angle_rad))-Seat_stay_junction0
        y = BB_textfield/torch.tan(Seat_angle_rad)
        z = torch.sqrt(torch.clip((CS_textfield ** 2)-(BB_textfield ** 2), min=eps))
        h = z-y
        j = (h*torch.sin(Seat_angle_rad))
        
        mask = (Seat_tube_type == 1) # ==1 when using dataset, ==0 when using mixeddf
        #q_true = j - (Seat_tube_diameter / 2) + ((Wheel_cut-RD) / 2)
        q_true = j + 16 #changed from 19.1 for buffer
        q_false = 1000000000 
        q = torch.where(mask, q_true, q_false)
        
        return (Wheel_cut / 2) - q

class FootIntersectsFrontWheel(ValidationFunction):
    def friendly_name(self) -> str:
        return "Foot intersects front wheel"

    def variable_names(self) -> List[str]:
        return ["Stack", "Head tube length textfield", "Head tube lower extension2", "Head angle", "BB textfield", "DT Length", "FORK0R", "Wheel diameter front"]

    def validate(self, designs: torch.tensor, eps=1e-6) -> torch.tensor:
        Stack, Head_tube_length_textfield, Head_tube_lower_extension2, Head_angle, BB_textfield, DT_length, fork0r, WDF = designs[:, :len(self.variable_names())].T
        # Extract variables from the DataFrame
        HTL = Head_tube_length_textfield
        HTLX = Head_tube_lower_extension2
        HTA = (Head_angle * math.pi) / 180  # Convert degrees to radians
        BBD = BB_textfield
        #FTY = BBD - WDR / 2 + WDF / 2
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
        crank_length_plus_foot_extension = 268.5 #172.5 crank length + 96 foot length (pt in bike cad)
        pedal_centerline_offset = 120
        return  (wheel_radius)**2 - pedal_centerline_offset**2 - (FCD - crank_length_plus_foot_extension)**2
    
class CrankHitsGroundInLowestPosition(ValidationFunction):
    def friendly_name(self) -> str:
        return "Crank hits ground in lowest position"

    def variable_names(self) -> List[str]:
        return ["BB textfield", "Wheel diameter rear"]

    def validate(self, designs: torch.tensor) -> torch.tensor:
        BB_textfield, WDR = designs[:, :len(self.variable_names())].T
        # Extract variables from the DataFrame
        wheel_radius = WDR / 2
        crank_length = 187.5 #changed from 172.5, doesnt factor in pedal length (variables are A and P in bikecad)
        return  (crank_length + BB_textfield) - wheel_radius

class RGBvalueGreaterThan255(ValidationFunction): #less than 0 covered in PositiveValueNegative
    def friendly_name(self) -> str:
        return "RGB value greater than 255"

    def variable_names(self) -> List[str]:
        return ["FIRST color R_RGB", "FIRST color G_RGB", "FIRST color B_RGB"]

    def validate(self, designs: torch.tensor) -> torch.tensor:
        color_overflow = designs - 255
        overflow_clipped = torch.clip(color_overflow, min=0)
        total = torch.sum(overflow_clipped, dim=1)
        #return total if total > 0 else sum of color_overflow (required for calculation of default weights)
        mask = total > 0
        return total * mask.float() + color_overflow.sum(dim=1) * (1 - mask.float())

class ChainStaysIntersect(ValidationFunction):
    def friendly_name(self) -> str:
        return "Chain stays intersect"

    def variable_names(self) -> List[str]:
        return ["csd", "Chain stay position on BB","BB length"]

    def validate(self, designs: torch.tensor) -> torch.tensor:
        csd, csbb, bbl = designs[:, :len(self.variable_names())].T
        # Extract variables from the DataFrame
        return  ((csd/2) + csbb) - (bbl/2)    

class TubeWallThicknessExceedsRadius(ValidationFunction):
    def friendly_name(self) -> str:
        return "Tube wall thickness exceeds radius"

    def variable_names(self) -> List[str]:
        # diameter, wall‐thickness pairs for each tube:
        return [
            "ttd",                       "Wall thickness Top tube",
            "csd",                       "Wall thickness Chain stay",
            "ssd",                       "Wall thickness Seat stay",
            "dtd",                       "Wall thickness Down tube",
            "Seat tube diameter",        "Wall thickness Seat tube",
            "Head tube diameter",        "Wall thickness Head tube",
            "BB diameter",               "Wall thickness Bottom Bracket",
        ]

    def validate(self, designs: torch.Tensor) -> torch.Tensor:
        # pull out the first 14 cols (7 pairs), reshape to (batch, 7, 2)
        vals = designs[:, :len(self.variable_names())]
        pairs = vals.reshape(vals.shape[0], -1, 2)
        diameters  = pairs[:, :, 0]
        thickness  = pairs[:, :, 1]
        # compute per‐tube violation = thickness − (diameter/2)
        violation  = thickness - (diameters / 2)
        # sum any positives across all tubes
        return torch.sum(torch.clamp(violation, min=0), dim=1)

class DownTubeIntersectsFrontWheel(ValidationFunction):
    def friendly_name(self) -> str:
        return "Down tube intersects front wheel"

    def variable_names(self):
        # Same ordering as the math below
        return [
            "Stack", "Head tube length textfield", "Head tube lower extension2", "Head angle", "DT Length", "BB textfield", "FORK0R", "Wheel diameter front", "Wheel diameter rear", "Down tube diameter",
        ]

    def validate(self, designs: torch.Tensor, eps=1e-6) -> torch.tensor:
        (stack, htl, htlx, head_angle, dt_len, bb_drop, fork0r, wdf, wdr, dt_dia) = designs[:, :len(self.variable_names())].T

        theta = head_angle * math.pi / 180.0  # radians
        theta = torch.clip(theta, min=eps)

        DTJY = stack - ((htl-htlx)*torch.sin(theta))
        DTJX = torch.sqrt(torch.clip((dt_len**2)-(DTJY**2), min=0))

        FBBD = bb_drop - wdr/2 + wdf/2  # y coordinate of front wheel axle, relative to BB, BB is (0,0)

        Fork_L_plus_HTLX_y = DTJY - FBBD + fork0r * torch.cos(theta)  # Fork L plus HTLX y component
        Fork_L_plus_HTLX_plus_spacer = Fork_L_plus_HTLX_y / torch.sin(theta)  # Fork L plus HTLX 
        Fork_L_plus_HTLX_x = Fork_L_plus_HTLX_plus_spacer * torch.cos(theta)  # Fork L plus HTLX x component

        FWX = DTJX + Fork_L_plus_HTLX_x + fork0r * torch.sin(theta) # x coord of front wheel axle, realtive to BB

        #dist = torch.abs(DTJY*FWX - DTJX*FBBD) / torch.sqrt(DTJX**2 + DTJY**2)

        DTJ_angle = torch.atan2(DTJY, DTJX)
        FW_angle = torch.atan2(FBBD, FWX)
        DTJBBFW_angle = DTJ_angle-FW_angle

        FW_dist = torch.sqrt(FWX**2 + FBBD**2)
        shortest_dist = torch.sin(DTJBBFW_angle)*FW_dist
        
        wheel_radius = wdf / 2.0
        tube_radius  = dt_dia / 2.0
        

        # Positive result → intersection (invalid)
        return wheel_radius - (shortest_dist - tube_radius)

bike_bench_validation_functions: List[ValidationFunction] = [
    SaddleHeightTooSmall(),
    SeatPostTooShort(),
    HeadTubeLowerExtensionLongerThanHeadTube(),
    HeadTubeUpperExtensionAndLowerExtensionOverlap(),
    StrictlyPositiveParameterIsNegative(),
    ChainStaySmallerThanRearWheelRadius(),
    ChainStayShorterThanBBDrop(),
    SeatStaySmallerThanRearWheelRadius(),
    SeatTubeIntersectsRearWheel(),
    DownTubeCantReachHeadTube(),
    RearWheelCutoutSeversSeatTube(),
    FootIntersectsFrontWheel(),
    CrankHitsGroundInLowestPosition(),
    RGBvalueGreaterThan255(),
    ChainStaysIntersect(),
    TubeWallThicknessExceedsRadius(),
    DownTubeIntersectsFrontWheel(),
]

difficult_validation_functions: List[ValidationFunction] = [
    # SaddleHeightTooSmall(),
    # SeatPostTooShort(),
    # HeadTubeLowerExtensionLongerThanHeadTube(),
    # HeadTubeUpperExtensionAndLowerExtensionOverlap(),
    # StrictlyPositiveParameterIsNegative(),
    # ChainStaySmallerThanRearWheelRadius(),
    # ChainStayShorterThanBBDrop(),
    # SeatStaySmallerThanRearWheelRadius(),
    # SeatTubeIntersectsRearWheel(),
    # DownTubeCantReachHeadTube(),
    # RearWheelCutoutSeversSeatTube(),
    FootIntersectsFrontWheel(),
    # CrankHitsGroundInLowestPosition(),
    # RGBvalueGreaterThan255(),
    # ChainStaysIntersect(),
    TubeWallThicknessExceedsRadius(),
    # DownTubeIntersectsFrontWheel(),
]
