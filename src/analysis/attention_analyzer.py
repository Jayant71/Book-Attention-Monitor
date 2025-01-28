from typing import Dict, Any, Optional

class AttentionAnalyzer:
    YAW_THRESHOLD = 30.0
    PITCH_THRESHOLD = 20.0

    @staticmethod
    def analyze_attention(face_details: Dict[str, Any]) -> Dict[str, Any]:
        if not face_details.get('FaceDetails'):
            return {
                'is_attentive': False,
                'reason': 'No face detected',
                'confidence': 0
            }

        face = face_details['FaceDetails'][0]
        pose = face['Pose']
        yaw = abs(pose['Yaw'])
        pitch = abs(pose['Pitch'])

        is_attentive = yaw < AttentionAnalyzer.YAW_THRESHOLD and pitch < AttentionAnalyzer.PITCH_THRESHOLD

        return {
            'is_attentive': is_attentive,
            'reason': f'Head pose: Yaw={yaw:.1f}°, Pitch={pitch:.1f}°',
            'confidence': face['Confidence']
        }
