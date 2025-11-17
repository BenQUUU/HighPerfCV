#pragma once

void launchGrayscaleKernel(const unsigned char* d_input,
                             unsigned char* d_output,
                             int width, int height,
                             size_t input_pitch, size_t output_pitch);
