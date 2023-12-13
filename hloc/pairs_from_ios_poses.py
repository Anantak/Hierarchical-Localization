# pairs_from_ios_poses.py

import argparse
import numpy as np
import scipy.spatial

def read_poses(file_path):
    poses = {}
    image_counter = 1  # Start with 1 for naming images as 'out-1', 'out-2', etc.

    with open(file_path, 'r') as file:
        for line in file:
            # Skip comments
            if line.startswith('#'):
                continue

            # Parse the line
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue  # Ensure there are enough parts to form a valid pose

            # Extract relevant data
            tx, ty, tz = map(float, parts[2:5])
            qx, qy, qz, qw = map(float, parts[5:9])

            # Form the image name
            image_name = f'out-{image_counter}'
            image_counter += 1

            # Add to the dictionary
            poses[image_name] = {
                'tx': tx, 'ty': ty, 'tz': tz,
                'qx': qx, 'qy': qy, 'qz': qz, 'qw': qw
            }

    return poses


def quaternion_to_rotmat(qx, qy, qz, qw):
    # First, normalize the quaternion to ensure it represents a valid rotation
    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    # Compute the rotation matrix elements
    r11 = 1 - 2*qy**2 - 2*qz**2
    r12 = 2*qx*qy - 2*qz*qw
    r13 = 2*qx*qz + 2*qy*qw

    r21 = 2*qx*qy + 2*qz*qw
    r22 = 1 - 2*qx**2 - 2*qz**2
    r23 = 2*qy*qz - 2*qx*qw

    r31 = 2*qx*qz - 2*qy*qw
    r32 = 2*qy*qz + 2*qx*qw
    r33 = 1 - 2*qx**2 - 2*qy**2

    # Form the rotation matrix
    rotmat = np.array([
        [r11, r12, r13],
        [r21, r22, r23],
        [r31, r32, r33]
    ])

    return rotmat

def get_pairwise_distances(poses):
    ids = list(poses.keys())
    Rs = []  # Rotation matrices
    ts = []  # Translation vectors

    for id_ in ids:
        pose = poses[id_]
        R = quaternion_to_rotmat(pose['qx'], pose['qy'], pose['qz'], pose['qw'])
        t = np.array([pose['tx'], pose['ty'], pose['tz']])
        Rs.append(R)
        ts.append(t)

    Rs = np.stack(Rs, 0)
    ts = np.stack(ts, 0)

    # Convert poses to camera-to-world
    Rs = Rs.transpose(0, 2, 1)
    ts = -(Rs @ ts[:, :, None])[:, :, 0]

    # Calculate pairwise distances
    dist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(ts))

    # Calculate rotational differences
    axes = Rs[:, :, -1]  # Using last column as principal axis
    dots = np.einsum('mi,ni->mn', axes, axes)
    dR = np.rad2deg(np.arccos(np.clip(dots, -1., 1.)))

    return ids, dist, dR


def main(poses_file, output, num_matched, rotation_threshold):
    poses = read_poses(poses_file)
    ids, dist, dR = get_pairwise_distances(poses)

    # Select pairs based on distance and rotation threshold
    pairs = []
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            if dist[i, j] < rotation_threshold and dR[i, j] < rotation_threshold:
                # Append .jpg extension to the image names
                pairs.append((f'{ids[i]}.jpg', f'{ids[j]}.jpg'))

    # Limit the number of matched pairs if necessary
    pairs = pairs[:num_matched]

    # Write pairs to output file
    with open(output, 'w') as f:
        for pair in pairs:
            f.write(f'{pair[0]} {pair[1]}\n')

    print(f'Found {len(pairs)} pairs written to {output}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--poses_file', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--num_matched', required=True, type=int)
    parser.add_argument('--rotation_threshold', default=30, type=float)
    args = parser.parse_args()
    main(**vars(args))
