import React from 'react';
import { Modal, StyleSheet, View } from 'react-native';

import { FaceAuthView, type FaceAuthViewProps } from './FaceAuthView';

export interface FaceAuthModalProps extends Omit<FaceAuthViewProps, 'style'> {
  /** Controls modal visibility. */
  visible: boolean;
}

/**
 * Drop-in modal wrapper around FaceAuthView.
 *
 * Usage:
 * ```tsx
 * <FaceAuthModal
 *   visible={showAuth}
 *   mode="identify"
 *   models={models}
 *   onResult={result => {
 *     if (result.ok) console.log('Authenticated:', result.personnelId);
 *     setShowAuth(false);
 *   }}
 *   onClose={() => setShowAuth(false)}
 * />
 * ```
 */
export function FaceAuthModal({ visible, ...viewProps }: FaceAuthModalProps) {
  return (
    <Modal
      visible={visible}
      animationType="slide"
      statusBarTranslucent
      onRequestClose={viewProps.onClose}>
      <View style={styles.container}>
        <FaceAuthView {...viewProps} style={styles.view} />
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  view: { flex: 1 },
});
