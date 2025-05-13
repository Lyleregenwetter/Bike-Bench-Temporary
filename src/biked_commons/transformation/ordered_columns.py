#TODO: Rename an dmove to combined_representation

ORDERED_COLUMNS = ['CS textfield', 'BB textfield', 'Stack', 'Head angle', 'Head tube length textfield', 'Seat stay '
                                                                                                        'junction0',
                   'Seat tube length', 'Seat angle', 'DT Length', 'BB diameter', 'ttd', 'dtd', 'csd', 'ssd',
                   'Chain stay position on BB', 'SSTopZOFFSET', 'Head tube upper extension2', 'Seat tube '
                                                                                              'extension2',
                   'Head tube lower extension2', 'SEATSTAYbrdgshift', 'CHAINSTAYbrdgshift', 'SEATSTAYbrdgdia1',
                   'CHAINSTAYbrdgdia1', 'SEATSTAYbrdgCheck', 'CHAINSTAYbrdgCheck', 'Dropout spacing',
                   'Wall thickness Bottom Bracket', 'Wall thickness Top tube', 'Wall thickness Head tube',
                   'Wall thickness Down tube', 'Wall thickness Chain stay', 'Wall thickness Seat stay',
                   'Wall thickness Seat tube', 'ERD rear', 'Wheel width rear', 'BSD front', 'Wheel width front',
                   'ERD front', 'BSD rear', 'Display AEROBARS', 'BB length', 'Head tube diameter', 'Wheel cut',
                   'Seat tube diameter', 'Front Fender include', 'Rear Fender include', 'Number of cogs',
                   'Number of chainrings', 'Display RACK', 'FIRST color R_RGB', 'FIRST color G_RGB', 'FIRST color '
                                                                                                     'B_RGB',
                   'SPOKES composite front', 'SBLADEW front', 'SBLADEW rear', 'Saddle length', 'Saddle height',
                   'Down tube diameter', 'Seatpost LENGTH', 'MATERIAL OHCLASS: ALUMINIUM', 'MATERIAL OHCLASS: '
                                                                                           'BAMBOO',
                   'MATERIAL OHCLASS: CARBON', 'MATERIAL OHCLASS: OTHER', 'MATERIAL OHCLASS: STEEL',
                   'MATERIAL OHCLASS: TITANIUM', 'Dropout spacing style OHCLASS: 0', 'Dropout spacing style OHCLASS: '
                                                                                     '1', 'Dropout spacing style '
                                                                                          'OHCLASS: 2',
                   'Dropout spacing style OHCLASS: 3', 'Fork type OHCLASS: 0', 'Fork type OHCLASS: 1', 'Fork type '
                                                                                                       'OHCLASS: 2',
                   'Stem kind OHCLASS: 0', 'Stem kind OHCLASS: 1', 'Stem kind OHCLASS: 2', 'Handlebar style OHCLASS: '
                                                                                           '0', 'Handlebar style '
                                                                                                'OHCLASS: 1',
                   'Handlebar style OHCLASS: 2', 'Head tube type OHCLASS: 0', 'Head tube type OHCLASS: 1',
                   'Head tube type OHCLASS: 2', 'Head tube type OHCLASS: 3', 'Top tube type OHCLASS: 0',
                   'Top tube type OHCLASS: 1', 'bottle SEATTUBE0 show OHCLASS: False', 'bottle SEATTUBE0 show '
                                                                                       'OHCLASS: True',
                   'bottle DOWNTUBE0 show OHCLASS: False', 'bottle DOWNTUBE0 show OHCLASS: True', 'BELTorCHAIN '
                                                                                                  'OHCLASS: 0',
                   'BELTorCHAIN OHCLASS: 1', 'RIM_STYLE front OHCLASS: DISC', 'RIM_STYLE front OHCLASS: SPOKED',
                   'RIM_STYLE front OHCLASS: TRISPOKE', 'RIM_STYLE rear OHCLASS: DISC', 'RIM_STYLE rear OHCLASS: '
                                                                                        'SPOKED', 'RIM_STYLE rear '
                                                                                                  'OHCLASS: TRISPOKE']
bike_bench_columns = ['CS textfield', 'BB textfield', 'Stack', 'Head angle',
       'Head tube length textfield', 'Seat stay junction0', 'Seat tube length',
       'Seat angle', 'DT Length', 'FORK0R', 'BB diameter', 'ttd', 'dtd', 'csd',
       'ssd', 'Chain stay position on BB', 'SSTopZOFFSET',
       'Head tube upper extension2', 'Seat tube extension2',
       'Head tube lower extension2', 'SEATSTAYbrdgshift', 'CHAINSTAYbrdgshift',
       'SEATSTAYbrdgdia1', 'CHAINSTAYbrdgdia1', 'SEATSTAYbrdgCheck',
       'CHAINSTAYbrdgCheck', 'Dropout spacing',
       'Wall thickness Bottom Bracket', 'Wall thickness Top tube',
       'Wall thickness Head tube', 'Wall thickness Down tube',
       'Wall thickness Chain stay', 'Wall thickness Seat stay',
       'Wall thickness Seat tube', 'Wheel diameter front', 'RDBSD',
       'Wheel diameter rear', 'FDBSD', 'Display AEROBARS', 'BB length',
       'Head tube diameter', 'Wheel cut', 'Seat tube diameter',
       'bottle SEATTUBE0 show', 'bottle DOWNTUBE0 show',
       'Front Fender include', 'Rear Fender include', 'BELTorCHAIN',
       'Number of cogs', 'Number of chainrings', 'Display RACK',
       'FIRST color R_RGB', 'FIRST color G_RGB', 'FIRST color B_RGB',
       'SPOKES composite front', 'SPOKES composite rear', 'SBLADEW front',
       'SBLADEW rear', 'Saddle length', 'Saddle height', 'Down tube diameter',
       'Seatpost LENGTH', 'MATERIAL OHCLASS: ALUMINIUM',
       'MATERIAL OHCLASS: BAMBOO', 'MATERIAL OHCLASS: CARBON',
       'MATERIAL OHCLASS: OTHER', 'MATERIAL OHCLASS: STEEL',
       'MATERIAL OHCLASS: TITANIUM', 'Head tube type OHCLASS: 0',
       'Head tube type OHCLASS: 1', 'Head tube type OHCLASS: 2',
       'Head tube type OHCLASS: 3', 'RIM_STYLE front OHCLASS: DISC',
       'RIM_STYLE front OHCLASS: SPOKED', 'RIM_STYLE front OHCLASS: TRISPOKE',
       'RIM_STYLE rear OHCLASS: DISC', 'RIM_STYLE rear OHCLASS: SPOKED',
       'RIM_STYLE rear OHCLASS: TRISPOKE', 'Handlebar style OHCLASS: 0',
       'Handlebar style OHCLASS: 1', 'Handlebar style OHCLASS: 2',
       'Stem kind OHCLASS: 0', 'Stem kind OHCLASS: 1', 'Stem kind OHCLASS: 2',
       'Fork type OHCLASS: 0', 'Fork type OHCLASS: 1', 'Fork type OHCLASS: 2',
       'Seat tube type OHCLASS: 0', 'Seat tube type OHCLASS: 1',
       'Seat tube type OHCLASS: 2']

oh_columns = [
            [
                "MATERIAL OHCLASS: ALUMINIUM",
                "MATERIAL OHCLASS: BAMBOO",
                "MATERIAL OHCLASS: CARBON",
                "MATERIAL OHCLASS: OTHER",
                "MATERIAL OHCLASS: STEEL",
                "MATERIAL OHCLASS: TITANIUM",
            ],
            [
                "Head tube type OHCLASS: 0",
                "Head tube type OHCLASS: 1",
                "Head tube type OHCLASS: 2",
                "Head tube type OHCLASS: 3",
            ],
            [
                "RIM_STYLE front OHCLASS: DISC",
                "RIM_STYLE front OHCLASS: SPOKED",
                "RIM_STYLE front OHCLASS: TRISPOKE",
            ],
            [
                "RIM_STYLE rear OHCLASS: DISC",
                "RIM_STYLE rear OHCLASS: SPOKED",
                "RIM_STYLE rear OHCLASS: TRISPOKE",
            ],
            [
                "Handlebar style OHCLASS: 0",
                "Handlebar style OHCLASS: 1",
                "Handlebar style OHCLASS: 2",
            ],
            [
                "Stem kind OHCLASS: 0",
                "Stem kind OHCLASS: 1",
                "Stem kind OHCLASS: 2",
            ],
            [
                "Fork type OHCLASS: 0",
                "Fork type OHCLASS: 1",
                "Fork type OHCLASS: 2",
            ],
            [
                "Seat tube type OHCLASS: 0",
                "Seat tube type OHCLASS: 1",
                "Seat tube type OHCLASS: 2",
            ]
        ]


oh_bool_columns = [
    "MATERIAL OHCLASS: ALUMINIUM",
    "MATERIAL OHCLASS: BAMBOO",
    "MATERIAL OHCLASS: CARBON",
    "MATERIAL OHCLASS: OTHER",
    "MATERIAL OHCLASS: STEEL",
    "MATERIAL OHCLASS: TITANIUM",

    "Head tube type OHCLASS: 0",
    "Head tube type OHCLASS: 1",
    "Head tube type OHCLASS: 2",
    "Head tube type OHCLASS: 3",

    "RIM_STYLE front OHCLASS: DISC",
    "RIM_STYLE front OHCLASS: SPOKED",
    "RIM_STYLE front OHCLASS: TRISPOKE",

    "RIM_STYLE rear OHCLASS: DISC",
    "RIM_STYLE rear OHCLASS: SPOKED",
    "RIM_STYLE rear OHCLASS: TRISPOKE",

    "Handlebar style OHCLASS: 0",
    "Handlebar style OHCLASS: 1",
    "Handlebar style OHCLASS: 2",

    "Stem kind OHCLASS: 0",
    "Stem kind OHCLASS: 1",
    "Stem kind OHCLASS: 2",

    "Fork type OHCLASS: 0",
    "Fork type OHCLASS: 1",
    "Fork type OHCLASS: 2",

    "Seat tube type OHCLASS: 0",
    "Seat tube type OHCLASS: 1",
    "Seat tube type OHCLASS: 2",

    'bottle SEATTUBE0 show',
    'bottle DOWNTUBE0 show',
    'BELTorCHAIN',
    'SEATSTAYbrdgCheck',
    'CHAINSTAYbrdgCheck',
]

USABILITY_COLUMNS = ['Saddle height', 'Stack', 'CS textfield']