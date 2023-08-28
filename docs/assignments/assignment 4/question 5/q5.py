fiducial_points = {
    'L-prime': 8.6,
    'P': 8.9,
    'P-prime': 9.1,
    'Q': 9.3,
    'R': 9.5,
    'S': 9.6,
    'S-prime': 9.9,
    'T': 10.4,
    'T-prime': 10.6
}

features = {
    'RQ': fiducial_points['R'] - fiducial_points['Q'],
    'RP-prime': fiducial_points['R'] - fiducial_points['P-prime'],
    'RP': fiducial_points['R'] - fiducial_points['P'],
    'RL-prime': fiducial_points['R'] - fiducial_points['L-prime'],
    'RS': fiducial_points['R'] - fiducial_points['S'],
    'RS-prime': fiducial_points['R'] - fiducial_points['S-prime'],
    'RT': fiducial_points['R'] - fiducial_points['T'],
    'RT-prime': fiducial_points['R'] - fiducial_points['T-prime']
}

for feature, value in features.items():
    print(f"{feature} = {value:.2f} s")
