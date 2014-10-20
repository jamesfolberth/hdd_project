function [r] = isOctave()
%
% Function to test if being run in GNU Octave or MATLAB
% return r~=0 if octave; return r=0 if not octave

   persistent x;
   if ( isempty(x) )
      x = exist('OCTAVE_VERSION','builtin'); % ~=0 if octave, =0 if MATLAB 
   end
   r = x;
end
