/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  // Create a random number generator
  default_random_engine gen;

  // Set the number of particles for our particle filter
  num_particles = 1000;

  // Standard deviations for x, y, and theta
  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];

  // Create a normal (Gaussian) distribution for x, y, & theta
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  // Generate the initial particles
  for (int i = 0; i < num_particles; ++i) {
    // Create a particle with x, y & theta sampled from the Gaussian distributions above
    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;

    // Add the particle to the list
    particles.push_back(particle);

    // initialize all weights to 1.0
    weights.push_back(1.0);
  }

  // Set the is_initialized flag to true
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  // Create a random number generator
  default_random_engine gen;

  // Standard deviations for x, y, and theta
  double std_x = std_pos[0];
  double std_y = std_pos[1];
  double std_theta = std_pos[2];

  // precalculate some values to reduce the work done in the loop below
  double c1 = velocity * delta_t;
  double c2 = yaw_rate * delta_t;

  if(yaw_rate != 0) {
    c1 = velocity/yaw_rate;
  }

  for (int i = 0; i < num_particles; ++i) {
    // use the motion model to predict the position and yaw at t+1
    if(yaw_rate == 0) {
      particles[i].x += c1 * cos(particles[i].theta);
      particles[i].y += c1 * sin(particles[i].theta);
      // particles[i].theta stays the same
    } else {
      particles[i].x += c1 * (sin(particles[i].theta + c2) - sin(particles[i].theta));
      particles[i].y += c1 * (cos(particles[i].theta) - cos(particles[i].theta + c2));
      particles[i].theta += c2;
    }

    // Create a normal (Gaussian) distribution for each particle's x, y, & theta
    normal_distribution<double> dist_x(particles[i].x, std_x);
    normal_distribution<double> dist_y(particles[i].y, std_y);
    normal_distribution<double> dist_theta(particles[i].theta, std_theta);

    // sample from the distribution to add the Gaussian noise
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  for (int i = 0; i < num_particles; ++i) {

    // Initialize vector to hold landmarks that are predicted to be within sensor_range
    std::vector<LandmarkObs> predicted;

    // Create a vector of landmarks that are predicted to be in range of our sensors
    for(const auto landmark : map_landmarks.landmark_list) {
      // Calculate the expected distance to each landmark
      double dist_to_landmark = dist(
          landmark.x_f, landmark.y_f, particles[i].x, particles[i].y);

      // If the expected distance falls within the sensor_range, add it to the vector of predicted landmarks
      if(dist_to_landmark < sensor_range) {
        LandmarkObs pred;
        pred.x = landmark.x_f;
        pred.y = landmark.y_f;
        pred.id = landmark.id_i;

        predicted.push_back(pred);
      }
    }

    // clear the associated landmarks to the particle
    particles[i].associations.clear();
    particles[i].sense_x.clear();
    particles[i].sense_y.clear();

    // Multivariate-Gaussian probability density vector
    vector<double> mgpd(observations.size());

    // Convert each observation from the vehicle's coordinate system to
    // the map's coordinate system using a homogeneous transformation.
    for (size_t j = 0; j < observations.size(); ++j) {
      // homogeneously transformed observation coordinates
      double x = particles[i].x + (cos(particles[i].theta) * observations[j].x) - (sin(particles[i].theta) * observations[j].y);
      double y = particles[i].y + (sin(particles[i].theta) * observations[j].x) + (cos(particles[i].theta) * observations[j].y);

      // sort the predicted landmarks by their distance to the current transformed observed landmark
      sort(predicted.begin(), predicted.end(),
           [&](LandmarkObs j, LandmarkObs k) {
             return dist(x, y, j.x, j.y) <
                    dist(x, y, k.x, k.y);
           });

      // the first element should be the nearest
      LandmarkObs nearest = predicted[0];

      // add the data from the nearest landmark (to current obs) to the particle's associations
      particles[i].associations.push_back(nearest.id);
      particles[i].sense_x.push_back(nearest.x);
      particles[i].sense_y.push_back(nearest.y);

      // calculate the error between the observed landmark and the predicted landmark
      double dx = x - nearest.x;
      double dy = y - nearest.y;

      // calculate the weight as a product of the multivariate gaussian probability densities for all observed landmarks
      mgpd[j] = (exp(-((dx * dx) / (2 * std_landmark[0] * std_landmark[0]) +
                       (dy * dy) / (2 * std_landmark[1] * std_landmark[1]))) /
                 (2 * M_PI * std_landmark[0] * std_landmark[1]));
    }

    // update the weight vector with the final product of multivariate gaussian probability densities
    weights[i] = particles[i].weight = accumulate(mgpd.begin(), mgpd.end(), 1.0, multiplies<double>());

  }
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  // Create a random number generator
  default_random_engine gen;
  discrete_distribution<> d(weights.begin(), weights.end());

  // create a temporary vector of particles with size = particles.size()
  std::vector<Particle> temp_particles(particles.size());
  for (int i = 0; i < num_particles; ++i) {
    // for each new particle, sample from the previous vector of
    // particles, proportional to the weights
    temp_particles[i] = particles[d(gen)];
  }

  // replace the old set of particles with the resampled set
  particles = temp_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
