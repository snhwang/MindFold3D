/**
 * MindFold3D Optimized Rotation Controller
 * This module provides a more controlled and predictable rotation system for 3D shapes
 * that considers shape properties like rotation_confusability.
 */

class RotationController {
  constructor(options = {}) {
    this.options = {
      // How fast shapes rotate (radians per frame)
      baseRotationSpeed: options.baseRotationSpeed || 0.01,
      // How long to maintain a particular rotation axis before changing (ms)
      directionChangeInterval: options.directionChangeInterval || 3000,
      // Whether to auto-rotate when not being controlled by user
      autoRotate: options.autoRotate !== undefined ? options.autoRotate : true,
      // Confusability settings affect rotation behavior
      respectConfusability: options.respectConfusability !== undefined ? options.respectConfusability : true,
      // Time to pause after a full rotation (ms)
      pauseInterval: options.pauseInterval || 1000,
      // Whether to randomize initial rotation
      randomizeInitialRotation: options.randomizeInitialRotation !== undefined ? options.randomizeInitialRotation : true,
      // Initial rotation axis if not randomized
      initialAxis: options.initialAxis || { x: 0, y: 1, z: 0 }
    };

    // Current state
    this.currentRotationAxis = { ...this.options.initialAxis };
    this.lastDirectionChangeTime = Date.now();
    this.rotationSpeed = this.options.baseRotationSpeed;
    this.isPaused = false;
    this.pauseStartTime = 0;
    this.rotationDirection = 1; // 1 or -1
    this.rotationConfusability = "medium"; // Default
    this.userInteracting = false;
    this.completedRotations = 0;
    this.lastRotationTime = 0;
    
    // Track accumulated rotation for each axis
    this.accumulatedRotation = { x: 0, y: 0, z: 0 };
    
    // Initialize with random axis if specified
    if (this.options.randomizeInitialRotation) {
      this.randomizeRotationAxis();
    }
  }

  /**
   * Set the rotation confusability value for a shape
   * @param {string|number} confusability - "low", "medium", "high" or numeric value 0-1
   */
  setRotationConfusability(confusability) {
    this.rotationConfusability = confusability;
    
    // Adjust rotation parameters based on confusability
    if (this.options.respectConfusability) {
      if (confusability === "low" || (typeof confusability === "number" && confusability < 0.3)) {
        // Low confusability: rotation can be faster and more varied
        this.rotationSpeed = this.options.baseRotationSpeed * 1.5;
        this.options.directionChangeInterval = 3000;
      } else if (confusability === "high" || (typeof confusability === "number" && confusability > 0.7)) {
        // High confusability: rotation should be slower and more stable
        this.rotationSpeed = this.options.baseRotationSpeed * 0.7;
        this.options.directionChangeInterval = 5000;
      } else {
        // Medium confusability: balanced rotation
        this.rotationSpeed = this.options.baseRotationSpeed;
        this.options.directionChangeInterval = 4000;
      }
    }
  }

  /**
   * Set a specific rotation axis
   * @param {Object} axis - {x, y, z} values between 0-1
   */
  setRotationAxis(axis) {
    this.currentRotationAxis = { ...axis };
    this.lastDirectionChangeTime = Date.now();
  }

  /**
   * Choose a random rotation axis
   * Considers rotation_confusability to determine axis complexity
   */
  randomizeRotationAxis() {
    // For high confusability, prefer simpler rotation axes
    if (this.rotationConfusability === "high" || (typeof this.rotationConfusability === "number" && this.rotationConfusability > 0.7)) {
      // Prefer rotation around a single cardinal axis
      const axisChoice = Math.floor(Math.random() * 3);
      
      switch (axisChoice) {
        case 0: // X axis
          this.currentRotationAxis = { x: 1, y: 0, z: 0 };
          break;
        case 1: // Y axis (most natural for human perception)
          this.currentRotationAxis = { x: 0, y: 1, z: 0 };
          break;
        case 2: // Z axis
          this.currentRotationAxis = { x: 0, y: 0, z: 1 };
          break;
      }
    } else {
      // For medium/low confusability, allow more complex rotation axes
      const axisChoice = Math.floor(Math.random() * 7);
      
      switch (axisChoice) {
        case 0: // X axis
          this.currentRotationAxis = { x: 1, y: 0, z: 0 };
          break;
        case 1: // Y axis
          this.currentRotationAxis = { x: 0, y: 1, z: 0 };
          break;
        case 2: // Z axis
          this.currentRotationAxis = { x: 0, y: 0, z: 1 };
          break;
        case 3: // XY diagonal
          this.currentRotationAxis = { x: 1, y: 1, z: 0 };
          break;
        case 4: // XZ diagonal
          this.currentRotationAxis = { x: 1, y: 0, z: 1 };
          break;
        case 5: // YZ diagonal
          this.currentRotationAxis = { x: 0, y: 1, z: 1 };
          break;
        case 6: // XYZ diagonal
          this.currentRotationAxis = { x: 1, y: 1, z: 1 };
          break;
      }
    }
    
    // Normalize the vector to ensure consistent rotation speed
    this.normalizeRotationAxis();
    
    // Sometimes reverse direction
    this.rotationDirection = Math.random() > 0.5 ? 1 : -1;
  }
  
  /**
   * Normalize the rotation axis vector
   */
  normalizeRotationAxis() {
    const length = Math.sqrt(
      this.currentRotationAxis.x * this.currentRotationAxis.x + 
      this.currentRotationAxis.y * this.currentRotationAxis.y + 
      this.currentRotationAxis.z * this.currentRotationAxis.z
    );
    
    if (length > 0) {
      this.currentRotationAxis.x /= length;
      this.currentRotationAxis.y /= length;
      this.currentRotationAxis.z /= length;
    }
  }
  
  /**
   * Set user interaction state
   * @param {boolean} isInteracting - Whether the user is currently interacting with the shape
   */
  setUserInteracting(isInteracting) {
    this.userInteracting = isInteracting;
    
    // Reset accumulated rotation when user starts interacting
    if (isInteracting) {
      this.accumulatedRotation = { x: 0, y: 0, z: 0 };
    }
  }
  
  /**
   * Apply rotation to a THREE.js Object3D
   * @param {Object} object - THREE.js Object3D to rotate
   * @returns {boolean} - Whether rotation was applied
   */
  update(object) {
    if (!object || this.userInteracting) {
      return false;
    }
    
    const now = Date.now();
    
    // Handle pause between rotation cycles
    if (this.isPaused) {
      if (now - this.pauseStartTime > this.options.pauseInterval) {
        this.isPaused = false;
        this.randomizeRotationAxis();
      } else {
        return false; // Still paused
      }
    }
    
    // Check if we should change rotation axis
    if (now - this.lastDirectionChangeTime > this.options.directionChangeInterval) {
      // Check if we've completed a full rotation (approximately 2π radians)
      const totalRotation = Math.abs(this.accumulatedRotation.x) + 
                            Math.abs(this.accumulatedRotation.y) + 
                            Math.abs(this.accumulatedRotation.z);
                            
      if (totalRotation >= Math.PI * 2) {
        // Reset accumulated rotation
        this.accumulatedRotation = { x: 0, y: 0, z: 0 };
        this.completedRotations++;
        
        // For high confusability shapes, pause briefly after a full rotation
        // to allow the user to better perceive the shape
        if (this.rotationConfusability === "high" || (typeof this.rotationConfusability === "number" && this.rotationConfusability > 0.7)) {
          this.isPaused = true;
          this.pauseStartTime = now;
          return false;
        }
      }
      
      this.randomizeRotationAxis();
      this.lastDirectionChangeTime = now;
    }
    
    if (this.options.autoRotate) {
      // Apply the rotation
      const frameRotation = this.rotationSpeed * this.rotationDirection;
      
      object.rotation.x += this.currentRotationAxis.x * frameRotation;
      object.rotation.y += this.currentRotationAxis.y * frameRotation;
      object.rotation.z += this.currentRotationAxis.z * frameRotation;
      
      // Track accumulated rotation
      this.accumulatedRotation.x += this.currentRotationAxis.x * frameRotation;
      this.accumulatedRotation.y += this.currentRotationAxis.y * frameRotation;
      this.accumulatedRotation.z += this.currentRotationAxis.z * frameRotation;
      
      this.lastRotationTime = now;
      return true;
    }
    
    return false;
  }
}

// Export the controller class
export { RotationController }; 