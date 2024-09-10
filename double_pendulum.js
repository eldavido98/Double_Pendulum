"use strict";

/* Physical Constants (physical simulation) */
const G = 9.81 // gravitational acceleration
const M = 10; // mass
const L = 1.0; // length
const dt_Max = 10; // largest interval of time that the simulation will advance in a single step
const tailLenghh_Max = 150; // maximum tail length

/* Geometric Constants (visualisation) */
const barWidth = 0.045;
const barLength = 0.2;
const sunRadius = 0.1;
const pivotRadius = 0.025;
const massRadius = 0.04
const tailThickness = 0.013;

/* WebGL stuff */
const quad = new Float32Array([-1, -1, +1, -1, -1, +1, +1, +1]);

const shadShader = {
    vert: `
        attribute vec2 a_point;
        uniform vec2 u_aspect;
        varying vec2 v_point;
        void main() {
            v_point = a_point;
            gl_Position = vec4(a_point * ${massRadius*10} / u_aspect, 0, 1);
        }`,
    frag: `
        uniform vec2 u_aspect;
        uniform vec3 v_color;
        varying vec2 v_point;
        void main() {
            float dist = distance(vec2(0, 0), v_point);
            float v = smoothstep(1.0, 0.9, dist);
            //gl_FragColor = vec4(v_color, v);
            gl_FragColor = vec4(0, 0, 0.25, 0.5);
        }`
};
const sunShader = {
    vert: `
        attribute vec2 a_point;
        uniform vec2 u_center;
        uniform vec2 u_aspect;
        varying vec2 v_point;
        void main() {
            v_point = a_point;
            gl_Position = vec4(a_point * ${sunRadius} / u_aspect + u_center, 0, 1);
        }`,
    frag: `
        uniform vec2 u_aspect;
        uniform vec3 u_color;
        varying vec2 v_point;
        void main() {
            float dist = distance(vec2(0, 0), v_point);
            float v = smoothstep(1.0, 0.9, dist);
            gl_FragColor = vec4(u_color, v);
        }`
};
const pivotShader = {
    vert: `
        attribute vec2 a_point;
        uniform vec2 u_center;
        uniform vec2 u_aspect;
        varying vec2 v_point;
        void main() {
            v_point = a_point;
            gl_Position = vec4(a_point * ${pivotRadius} / u_aspect + u_center, 0, 1);
        }`,
    frag: `
        uniform vec2 u_aspect;
        uniform vec3 u_color;
        varying vec2 v_point;
        void main() {
            float dist = distance(vec2(0, 0), v_point);
            float v = smoothstep(1.0, 0.9, dist);
            gl_FragColor = vec4(u_color, v);
        }`
};
const massShader = {
    vert: `
        attribute vec2 a_point;
        uniform vec2 u_center;
        uniform vec2 u_aspect;
        varying vec2 v_point;
        void main() {
            v_point = a_point;
            gl_Position = vec4(a_point * ${massRadius} / u_aspect + u_center, 0, 1);
        }`,
    frag: `
        uniform vec2 u_aspect;
        uniform vec3 u_color;
        varying vec2 v_point;
        void main() {
            float dist = distance(vec2(0, 0), v_point);
            float v = smoothstep(1.0, 0.9, dist);
            gl_FragColor = vec4(u_color, v);
        }`
};
const barShader = {
    vert: `
        attribute vec2  a_point;
        uniform float u_angle;
        uniform vec2  u_attach;
        uniform vec2  u_aspect;
        void main() {
            mat2 rotate = mat2(+cos(u_angle), +sin(u_angle), -sin(u_angle), +cos(u_angle));
            vec2 pos = rotate * (a_point * vec2(1, ${barWidth}) + vec2(1, 0));
            gl_Position = vec4((pos * ${barLength} / u_aspect + u_attach), 0, 1);
        }`,
    frag: `
        uniform vec3 u_color;
        void main() {
            gl_FragColor = vec4(u_color, 1);
        }`
};
const tailShader = {
    vert: `
        attribute vec2  a_point;
        attribute float a_alpha;
        uniform vec2  u_aspect;
        varying float v_alpha;
        void main() {
            v_alpha = a_alpha;
            gl_Position = vec4(a_point * vec2(1, -1) / u_aspect, 0, 1);
        }`,
    frag: `
        uniform vec3  u_color;
        uniform float u_cutoff;
        varying float v_alpha;
        void main() {
            float icutoff = 1.0 - u_cutoff;
            gl_FragColor = vec4(u_color, max(0.0, v_alpha - u_cutoff) / icutoff);
        }`
};

/* Operators */
function normalize(a, b)
{
    let norm_2 = Math.sqrt(a * a + b * b);
    return [a / norm_2, b / norm_2];
}
function vector_difference(a_x, a_y, b_x, b_y)
{
    return [a_x - b_x, a_y - b_y];
}
// Compute the coordinates of the intersection between 2 lines
function line_intersection(l1_start, l1_end, l2_start, l2_end)
{
  let x, y;
  let m1, q1, m2, q2;

  const x1 = l1_start[0];
  const y1 = l1_start[1];
  const x2 = l1_end[0];
  const y2 = l1_end[1];
  const x3 = l2_start[0];
  const y3 = l2_start[1];
  const x4 = l2_end[0];
  const y4 = l2_end[1];

  if (x1 == x2) {
    // First line is vertical
    if (x3 == x4) {
      // Also second line is vertical: they are parallel!
      return null;
    }
    m2 = (y4 - y3) / (x4 - x3);
    q2 = ((x4 * y3) - (x3 * y4)) / (x4 - x3);
    x = x1;
    y = (m2 * x) + q2;
    return [x, y];
  }
  else if (x3 == x4) {
    // Second line is vertical
    if (x1 == x2) {
      // Also first line is vertical: they are parallel!
      return null;
    }
    m2 = (y2 - y1) / (x2 - x1);
    q2 = ((x2 * y1) - (x1 * y2)) / (x2 - x1);
    x = x3;
    y = (m2 * x) + q2;
    return [x, y];
  }
  else if (y1 == y2) {
    // First line is horizontal
    if (y3 == y4) {
      // Also second line is horizontal, they are parallel
      return null;
    }
    m2 = (y4 - y3) / (x4 - x3);
    q2 = ((x4 * y3) - (x3 * y4)) / (x4 - x3);
    y = y1;
    x = (y - q2) / m2;
    return [x, y];
  }
  else if (y3 == y4) {
    // Second line is horizontal
    if (y1 == y2) {
      // Also first line is horizontal, they are parallel
      return null;
    }
    m2 = (y2 - y1) / (x2 - x1);
    q2 = ((x2 * y1) - (x1 * y2)) / (x2 - x1);
    y = y3;
    x = (y - q2) / m2;
    return [x, y];
  }
  
  // Calculate slopes
  m1 = (y2 - y1) / (x2 - x1);
  m2 = (y4 - y3) / (x4 - x3);
  q1 = y1 - (x1 * ((y2 - y1) / (x2 - x1)));
  q2 = y3 - (x3 * ((y4 - y3) / (x4 - x3)));
  if (m1 == m2) {return null;}
  x = (q2 - q1) / (m1 - m2);
  y = ((m1 * q2) - (m2 * q1)) / (m1 - m2);
  
  return [x, y];
}
// Determine the position of a point with respect to a line
function point_position(point, l1_start, l1_end)
{
  let m, q;

  const [point_x, point_y] = point;
  const [x1, y1] = l1_start;
  const [x2, y2] = l1_end;

  if (x1 == x2) {
    // Line is vertical
    if (point_x > x1) {return +1}
    else if (point_x < x1) {return -1;}
    else {return 0;}
  }
  else if (y1 == y2) {
    // Line is horizontal
    if (point_y > y1) {return +1}
    else if (point_y < y1) {return -1;}
    else {return 0;}
  }
  else {
    m = (y2 - y1) / (x2 - x1);
    q = y1 - (x1 * ((y2 - y1) / (x2 - x1)));
    return Math.sign(m*point_x - point_y + q);
  }
}
// Given 3 points, it determines which one is on the left, which one is on the right and which one is on the center
function min_middle_max_X(points) {
    let point;
    let min_point = points[0];
    let middle_point = points[0];
    let max_point = points[0];

    for (let i = 1; i < points.length; i++) {
        point = points[i];
        if (point[0] < min_point[0]) {
            middle_point = min_point;
            min_point = point;}
        else if (point[0] > max_point[0]) {
            middle_point = max_point;
            max_point = point;}
        else {middle_point = point;}
    }
    return [min_point, middle_point, max_point];
}
// Determine if 4 points are on the same line
function points_on_line(pivot_pos, sun_pos, m1_pos, m2_pos) {
    let m, q, x = 0;
    const x1 = pivot_pos[0];
    const y1 = pivot_pos[1];
    const x2 = sun_pos[0];
    const y2 = sun_pos[1];
    const x3 = m1_pos[0];
    const y3 = m1_pos[1];
    const x4 = m2_pos[0];
    const y4 = m2_pos[1];

    if (x1 == x2) {
        x = x1;
        if ((Math.abs(x3 - x) < 0.0005) && (Math.abs(x4 - x) < 0.0005)) {return true;}
        else {return false}
    }
    else if (y1 == y2) {
        m = 0;
        q = y1;
    }
    else {
        m = (y2 - y1) / (x2 - x1);
        q = y1 - (x1 * ((y2 - y1) / (x2 - x1)));
    }
    if ((Math.abs(m*x3 + q - y3) < 0.005) && (Math.abs(m*x4 + q - y4) < 0.005)) {return true;}
    else {return false}
}
// Given 4 points, determine which one ('pivot' and 'm2') is furthest from point 'sun';
// if they are equidistant, it determines if is 'm1' furthest from point 'sun' or not
function points_distance(sun, pivot, m2, m1) {
    let dx, dy;
    let sun_pivot, sun_m2, sun_m1;

    dx = sun[0] - pivot[0];
    dy = sun[1] - pivot[1];
    sun_pivot = Math.sqrt(dx*dx + dy*dy);
    dx = sun[0] - m2[0];
    dy = sun[1] - m2[1];
    sun_m2 = Math.sqrt(dx*dx + dy*dy);
    if (sun_pivot > sun_m2) {
        return 0;
    }
    else if (sun_pivot < sun_m2) {
        return 2;
    }
    else {
        // Points 'pivot' and 'm2' are equidistant from point 'sun'
        dx = sun[0] - m1[0];
        dy = sun[1] - m1[1];
        sun_m1 = Math.sqrt(dx*dx + dy*dy);
        if (sun_pivot > sun_m1) {
            return 0;
        }
        else if (sun_pivot < sun_m1) {
            return 1;
        }
    }
}
// Compute the derivative of the 2 angles and of the 2 momentum of the 2 masses
function deriviative(a1, a2, p1, p2, damping_coeff)
{
    let c12 = Math.cos(a1 - a2);
    let s12 = Math.sin(a1 - a2);
    let s1 = Math.sin(a1);
    let s2 = Math.sin(a2);
    let a1_dot = (1 / (M * L * L)) * ((p1 - c12*p2) / (2 - c12*c12)) - a1*damping_coeff;
    let a2_dot = (1 / (M * L * L)) * ((2*p2 - c12*p1) / (2 - c12*c12)) - a2*damping_coeff;
    let p1_dot = (-2 * M * G * L * s1) + (-(p1 * p2 * s12)/((M * L * L) * (2 - c12*c12))) + (s12 * c12 * ((p1*p1 + 2*p2*p2 - 2*p1*p2*c12)/((M * L * L)*(1 + s12*s12 + s12*s12*s12*s12))));
    let p2_dot = (- M * G * L * s2) + ((p1 * p2 * s12)/((M * L * L) * (2 - c12*c12))) + (-s12 * c12 * ((p1*p1 + 2*p2*p2 - 2*p1*p2*c12)/((M * L * L)*(1 + s12*s12 + s12*s12*s12*s12))));
    return [a1_dot, a2_dot, p1_dot, p2_dot];
}

/* Update pendulum by timestep */
// Euler Method
function Euler_Method(a1, a2, p1, p2, dt, damping_coeff)
{
    let [a1_dot, a2_dot, p1_dot, p2_dot] = deriviative(a1, a2, p1, p2, damping_coeff);
    return [
        a1 + a1_dot * dt,
        a2 + a2_dot * dt,
        p1 + p1_dot * dt,
        p2 + p2_dot * dt
    ];
}
// Adams-Bashfort 3rd Order Method
function AdamsBashfort3rd_Method(a1_k, a2_k, p1_k, p2_k, a1_k1_dot, a2_k1_dot, p1_k1_dot, p2_k1_dot, a1_k2_dot, a2_k2_dot, p1_k2_dot, p2_k2_dot, dt, damping_coeff)
{
    let [a1_k_dot, a2_k_dot, p1_k_dot, p2_k_dot] = deriviative(a1_k, a2_k, p1_k, p2_k, damping_coeff);
    return [
        a1_k + (23*a1_k_dot - 16*a1_k1_dot + 5*a1_k2_dot)*dt/12,
        a2_k + (23*a2_k_dot - 16*a2_k1_dot + 5*a2_k2_dot)*dt/12,
        p1_k + (23*p1_k_dot - 16*p1_k1_dot + 5*p1_k2_dot)*dt/12,
        p2_k + (23*p2_k_dot - 16*p2_k1_dot + 5*p2_k2_dot)*dt/12,
        a1_k_dot, a2_k_dot, p1_k_dot, p2_k_dot
    ];
}
// Runge-Kutta 4th Order Method
function RungeKutta4th_Method(a1_k1, a2_k1, p1_k1, p2_k1, dt, damping_coeff)
{
    let [a1_k1_dot, a2_k1_dot, p1_k1_dot, p2_k1_dot] = deriviative(a1_k1, a2_k1, p1_k1, p2_k1, damping_coeff);
    let a1_k2 = a1_k1 + a1_k1_dot*dt/2;
    let a2_k2 = a2_k1 + a2_k1_dot*dt/2;
    let p1_k2 = p1_k1 + p1_k1_dot*dt/2;
    let p2_k2 = p2_k1 + p2_k1_dot*dt/2;

    let [a1_k2_dot, a2_k2_dot, p1_k2_dot, p2_k2_dot] = deriviative(a1_k2, a2_k2, p1_k2, p2_k2, damping_coeff);
    let a1_k3 = a1_k1 + a1_k2_dot*dt/2;
    let a2_k3 = a2_k1 + a2_k2_dot*dt/2;
    let p1_k3 = p1_k1 + p1_k2_dot*dt/2;
    let p2_k3 = p2_k1 + p2_k2_dot*dt/2;

    let [a1_k3_dot, a2_k3_dot, p1_k3_dot, p2_k3_dot] = deriviative(a1_k3, a2_k3, p1_k3, p2_k3, damping_coeff);
    let a1_k4 = a1_k1 + a1_k3_dot*dt;
    let a2_k4 = a2_k1 + a2_k3_dot*dt;
    let p1_k4 = p1_k1 + p1_k3_dot*dt;
    let p2_k4 = p2_k1 + p2_k3_dot*dt;

    let [a1_k4_dot, a2_k4_dot, p1_k4_dot, p2_k4_dot] = deriviative(a1_k4, a2_k4, p1_k4, p2_k4, damping_coeff);
    return [
        a1_k1 + (a1_k1_dot + 2*a1_k2_dot + 2*a1_k3_dot + a1_k4_dot)*dt/6,
        a2_k1 + (a2_k1_dot + 2*a2_k2_dot + 2*a2_k3_dot + a2_k4_dot)*dt/6,
        p1_k1 + (p1_k1_dot + 2*p1_k2_dot + 2*p1_k3_dot + p1_k4_dot)*dt/6,
        p2_k1 + (p2_k1_dot + 2*p2_k2_dot + 2*p2_k3_dot + p2_k4_dot)*dt/6
    ];
}

// Used to store (push) and return (visit) the tail's informations
function tail_history(tail_length)
{
    let h = {
        index: 0,
        length: 0,
        angle_values: new Float32Array(tail_length * 2),
        push: function(ang1, ang2, type)
        {
            if (type == 1)
            {
                h.angle_values[h.index*2 + 0] = Math.sin(ang1);
                h.angle_values[h.index*2 + 1] = Math.cos(ang1);
            }
            else if (type == 2)
            {
                h.angle_values[h.index*2 + 0] = Math.sin(ang1) + Math.sin(ang2);
                h.angle_values[h.index*2 + 1] = Math.cos(ang1) + Math.cos(ang2);
            }
            h.index = (h.index + 1) % tail_length;
            if (h.length < tail_length)
                h.length++;
        },
        visit: function(f)
        {
            for (let j = h.index + tail_length - 2; j > h.index + tail_length - h.length - 1; j--)
            {
                let a = (j + 1) % tail_length;
                let b = (j + 0) % tail_length;
                f(h.angle_values[a*2], h.angle_values[a*2 + 1], h.angle_values[b*2], h.angle_values[b*2 + 1]);
            }
        }
    };
    return h;
}

// Create a new double pendulum
function double_pendulum({
    TailColor = [0/255, 255/255, 0/255],
    DoublePendulumColor = [0/255, 0/255, 0/255],
    initial_conditions = null
} = {})
{
    let m1_tail = new tail_history(tailLenghh_Max);
    let m2_tail = new tail_history(tailLenghh_Max);
    let ang_bar1, ang_bar2, p1, p2;
    let [ang_bar1_k1_dot, ang_bar2_k1_dot, p1_k1_dot, p2_k1_dot, ang_bar1_k2_dot, ang_bar2_k2_dot, p1_k2_dot, p2_k2_dot] = [0, 0, 0, 0, 0, 0, 0, 0];

    let a1_dot, a2_dot, p1_dot, p2_dot;

    if (initial_conditions)
    {
        [ang_bar1, ang_bar2, p1, p2] = initial_conditions;
    }
    else
    {
        ang_bar1 = Math.random() * Math.PI / 2 + Math.PI * 3 / 4;         // ang_bar1 will be assigned a random angle in radians between '3*Pi/4' and '5*Pi/4' (135 - 225 [degrees])
        ang_bar2 = Math.random() * (2 * Math.PI);         // ang_bar2 will be assigned a random angle in radians between '0' and '2*Pi' (0 - 360 [degrees])
        p1 = 0.0;
        p2 = 0.0;
    }

    return {
        TailColor: TailColor,
        DoublePendulumColor: DoublePendulumColor,
        m1_tail: m1_tail,
        m2_tail: m2_tail,
        angles: function()
        {
            return [ang_bar1, ang_bar2];
        },
        momentum: function()
        {
            return [p1, p2];
        },
        positions: function()
        {
            let x1 = +Math.sin(ang_bar1);
            let y1 = -Math.cos(ang_bar1);
            let x2 = +Math.sin(ang_bar2) + x1;
            let y2 = -Math.cos(ang_bar2) + y1;
            return [x1, y1, x2, y2];
        },
        step: function(dt, solver_method, damping_coeff)
        {
            if (solver_method == 0) {[ang_bar1, ang_bar2, p1, p2] = RungeKutta4th_Method(ang_bar1, ang_bar2, p1, p2, dt, damping_coeff);}
            else if (solver_method == 1) {[ang_bar1, ang_bar2, p1, p2] = Euler_Method(ang_bar1, ang_bar2, p1, p2, dt, damping_coeff);}
            else if (solver_method == 2)
            {
                [ang_bar1, ang_bar2, p1, p2, a1_dot, a2_dot, p1_dot, p2_dot] = AdamsBashfort3rd_Method(ang_bar1, ang_bar2, p1, p2, ang_bar1_k1_dot, ang_bar2_k1_dot, p1_k1_dot, p2_k1_dot, ang_bar1_k2_dot, ang_bar2_k2_dot, p1_k2_dot, p2_k2_dot, dt, damping_coeff);
                [ang_bar1_k2_dot, ang_bar2_k2_dot, p1_k2_dot, p2_k2_dot] = [ang_bar1_k1_dot, ang_bar2_k1_dot, p1_k1_dot, p2_k1_dot];
                [ang_bar1_k1_dot, ang_bar2_k1_dot, p1_k1_dot, p2_k1_dot] = [a1_dot, a2_dot, p1_dot, p2_dot];
            }
            m1_tail.push(ang_bar1, ang_bar2, 1);
            m2_tail.push(ang_bar1, ang_bar2, 2);
        },
        manual_step_1: function(ang1)
        {
            ang_bar1 = ang1;
        },
        manual_step_2: function(ang2)
        {
            ang_bar2 = ang2;
        }
    };
}

/* Rendering */
// Convert tail line into a triangle strip
function polyline(hist, poly)
{
    const w = tailThickness;
    let i = -1;
    let x_0, y_0;
    let x_f, y_f;
    hist.visit(function(x1, y1, x2, y2)
    {
        if (++i == 0)
        {
            let [lx, ly] = vector_difference(x2, y2, x1, y1);
            let [nx, ny] = normalize(-ly, lx);
            poly[0] = x1 + w * nx;
            poly[1] = y1 + w * ny;
            poly[2] = x1 - w * nx;
            poly[3] = y1 - w * ny;
        }
        else
        {
            let [ax, ay] = vector_difference(x1, y1, x_0, y_0);
            [ax, ay] = normalize(ax, ay);
            let [bx, by] = vector_difference(x2, y2, x1, y1);
            [bx, by] = normalize(bx, by);
            let [tx, ty] = [ax + bx, ay + by];
            let [mx, my] = normalize(-ty, tx);
            let [lx, ly] = vector_difference(x1, y1, x_0, y_0);
            let [nx, ny] = normalize(-ly, lx);
            let len = Math.min(w, w/(mx*nx + my*ny));
            poly[i * 4 + 0] = x1 + mx * len;
            poly[i * 4 + 1] = y1 + my * len;
            poly[i * 4 + 2] = x1 - mx * len;
            poly[i * 4 + 3] = y1 - my * len;
        }
        x_0 = x1;
        y_0 = y1;
        x_f = x2;
        y_f = y2;
    });
    let [lx, ly] = vector_difference(x_f, y_f, x_0, y_0);
    let [nx, ny] = normalize(-ly, lx);
    i++;
    poly[i * 4 + 0] = x_f + w * nx;
    poly[i * 4 + 1] = y_f + w * ny;
    poly[i * 4 + 2] = x_f - w * nx;
    poly[i * 4 + 3] = y_f - w * ny;
}
// Render the bars, the tail, the pivot and the masses of a given double pendulum using WebGL based on the provided data and programs
function render_pend(gl, webgl, double_pendulum, tail_flag, m1_tail_flag, sun_position, shadow_flag)
{
    let diameter = barLength * 2;
    let width = gl.canvas.width;
    let height = gl.canvas.height;
    let smallest_dim = Math.min(width, height);
    let scaled_width = width / smallest_dim;
    let scaled_height = height / smallest_dim;

    let bar, bar1_x, bar1_y, bar2_x, bar2_y, ang1, ang2;
    let tail, cutoff_1, cutoff_2;
    let sun;
    let pivot;
    let mass, mass1_x, mass1_y, mass2_x, mass2_y;

    let sun_x, sun_y;
    let shadow1_x, shadow1_y, shadow2_x, shadow2_y;
    let triangle_1, triangle_2, triangle_3, triangle_4;
    let X_1, Y_1, X_2, Y_2, X_3, Y_3, X_4, Y_4;
    let shadow2_pos_0_1, sun_pos_0_1, sun_pos_1_2, zero_pos_1_2, aligned_flag;
    let left, center, right;
    let points_dist;
    
    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);

    // Bar
    bar = webgl.bar;
    gl.useProgram(bar.program);
    gl.uniform2f(bar.u_aspect, scaled_width, scaled_height);
    gl.bindBuffer(gl.ARRAY_BUFFER, webgl.quad);
    gl.enableVertexAttribArray(bar.a_point);
    gl.vertexAttribPointer(bar.a_point, 2, gl.FLOAT, false, 0, 0);
    [bar1_x, bar1_y, bar2_x, bar2_y] = double_pendulum.positions();
    [ang1, ang2] = double_pendulum.angles();
    bar1_x *= diameter / scaled_width;
    bar1_y *= diameter / scaled_height;
    gl.uniform3fv(bar.u_color, double_pendulum.DoublePendulumColor);
    gl.uniform2f(bar.u_attach, 0, 0);
    gl.uniform1f(bar.u_angle, (ang1 - Math.PI/2));
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.uniform2f(bar.u_attach, bar1_x, bar1_y);
    gl.uniform1f(bar.u_angle, (ang2 - Math.PI/2));
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    // Tail
    if (!tail_flag)
    {
        tail = webgl.tail;
        gl.useProgram(tail.program);
        gl.uniform2f(tail.u_aspect, scaled_width / diameter, scaled_height / diameter);
        gl.bindBuffer(gl.ARRAY_BUFFER, webgl.alpha);
        gl.enableVertexAttribArray(tail.a_alpha);
        gl.vertexAttribPointer(tail.a_alpha, 1, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, webgl.tailb);
        gl.enableVertexAttribArray(tail.a_point);
        gl.vertexAttribPointer(tail.a_point, 2, gl.FLOAT, false, 0, 0);
        if (double_pendulum.m2_tail.length || double_pendulum.m1_tail.length)
        {
            // Mass 2
            polyline(double_pendulum.m2_tail, webgl.tailpoly);
            gl.bufferSubData(gl.ARRAY_BUFFER, 0, webgl.tailpoly);
            gl.uniform3fv(tail.u_color, double_pendulum.TailColor);
            cutoff_1 = 1 - double_pendulum.m2_tail.length*2/double_pendulum.m2_tail.angle_values.length;
            gl.uniform1f(tail.u_cutoff, cutoff_1);
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, double_pendulum.m2_tail.length*2);
            // Mass 1
            if (m1_tail_flag)
            {
                polyline(double_pendulum.m1_tail, webgl.tailpoly);
                gl.bufferSubData(gl.ARRAY_BUFFER, 0, webgl.tailpoly);
                gl.uniform3fv(tail.u_color, double_pendulum.TailColor);
                cutoff_2 = 1 - double_pendulum.m1_tail.length*2/double_pendulum.m1_tail.angle_values.length;
                gl.uniform1f(tail.u_cutoff, cutoff_2);
                gl.drawArrays(gl.TRIANGLE_STRIP, 0, double_pendulum.m1_tail.length*2);
            }
        }
    }
    // Pivot
    pivot = webgl.pivot;
    gl.useProgram(pivot.program);
    gl.uniform2f(pivot.u_aspect, scaled_width, scaled_height);
    gl.bindBuffer(gl.ARRAY_BUFFER, webgl.quad);
    gl.enableVertexAttribArray(pivot.a_point);
    gl.vertexAttribPointer(pivot.a_point, 2, gl.FLOAT, false, 0, 0);
    gl.uniform3fv(pivot.u_color, double_pendulum.DoublePendulumColor);
    gl.uniform2f(pivot.u_center, 0, 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, quad.length);
    // Mass
    mass = webgl.mass;
    gl.useProgram(mass.program);
    gl.uniform2f(mass.u_aspect, scaled_width, scaled_height);
    gl.bindBuffer(gl.ARRAY_BUFFER, webgl.quad);
    gl.enableVertexAttribArray(mass.a_point);
    gl.vertexAttribPointer(mass.a_point, 2, gl.FLOAT, false, 0, 0);
    [mass1_x, mass1_y, mass2_x, mass2_y] = double_pendulum.positions();
    mass1_x *= diameter / scaled_width;
    mass1_y *= diameter / scaled_height;
    mass2_x *= diameter / scaled_width;
    mass2_y *= diameter / scaled_height;
    gl.uniform3fv(mass.u_color, double_pendulum.DoublePendulumColor);
    gl.uniform2f(mass.u_center, mass1_x, mass1_y);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, quad.length);
    gl.uniform2f(mass.u_center, mass2_x, mass2_y);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, quad.length);
    // Light
    sun = webgl.light;
    gl.useProgram(sun.program);
    gl.uniform2f(sun.u_aspect, scaled_width, scaled_height);
    gl.bindBuffer(gl.ARRAY_BUFFER, webgl.quad);
    gl.enableVertexAttribArray(sun.a_point);
    gl.vertexAttribPointer(sun.a_point, 2, gl.FLOAT, false, 0, 0);
    if (shadow_flag) {gl.uniform3fv(sun.u_color, [1, 0.843, 0]);    /* gold */}
    else {gl.uniform3fv(sun.u_color, [0.961,0.961,0.961]);  /* whitesmoke */}
    gl.uniform2f(sun.u_center, sun_position[0], sun_position[1]);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, quad.length);
    // Shadow
    [sun_x, sun_y] = [(sun_position[0] * 5.5), (sun_position[1] * 2.5)];
    left, center, right;
    points_dist;
    if (shadow_flag) {
        [shadow1_x, shadow1_y, shadow2_x, shadow2_y] = double_pendulum.positions();

        shadow1_x *= 5.5 * diameter / scaled_width;
        shadow1_y *= 2.5 * diameter / scaled_height;
        shadow2_x *= 5.5 * diameter / scaled_width;
        shadow2_y *= 2.5 * diameter / scaled_height;

        [X_1, Y_1] = line_intersection([-2.5, -2.5], [+2.5, -2.5], [sun_x, sun_y], [0, 0]);
        [X_2, Y_2] = line_intersection([-2.5, -2.5], [+2.5, -2.5], [sun_x, sun_y], [shadow1_x, shadow1_y]);
        [X_3, Y_3] = line_intersection([-2.5, -2.5], [+2.5, -2.5], [sun_x, sun_y], [shadow2_x, shadow2_y]);

        shadow2_pos_0_1 = point_position([shadow2_x, shadow2_y], [0, 0], [shadow1_x, shadow1_y]);
        sun_pos_0_1 = point_position([sun_x, sun_y], [0, 0], [shadow1_x, shadow1_y]);
        sun_pos_1_2 = point_position([sun_x, sun_y], [shadow1_x, shadow1_y], [shadow2_x, shadow2_y]);
        zero_pos_1_2 = point_position([0, 0], [shadow1_x, shadow1_y], [shadow2_x, shadow2_y]);
        aligned_flag = points_on_line([0, 0], [sun_x, sun_y], [shadow1_x, shadow1_y], [shadow2_x, shadow2_y]);

        [left, center, right] = min_middle_max_X([[X_1, Y_1], [X_2, Y_2], [X_3, Y_3]]);

        if ((X_1 == left[0]) && (Y_1 == left[1]) && (X_2 == center[0]) && (Y_2 == center[1]) && (X_3 == right[0]) && (Y_3 == right[1]) && (!aligned_flag)) {
            //console.log(0);
            triangle_1 = new Float32Array([0, 0, left[0], left[1], center[0], center[1]]);
            triangle_2 = new Float32Array([0, 0, shadow1_x, shadow1_y, center[0], center[1]]);
            triangle_3 = new Float32Array([shadow1_x, shadow1_y, center[0], center[1], right[0], right[1]]);
            triangle_4 = new Float32Array([shadow1_x, shadow1_y, shadow2_x, shadow2_y, right[0], right[1]]);
            draw_shadow(gl, webgl, triangle_1, scaled_width, scaled_height, 3);
            draw_shadow(gl, webgl, triangle_2, scaled_width, scaled_height, 3);
            draw_shadow(gl, webgl, triangle_3, scaled_width, scaled_height, 3);
            draw_shadow(gl, webgl, triangle_4, scaled_width, scaled_height, 3);
        }
        else if ((X_1 == right[0]) && (Y_1 == right[1]) && (X_2 == center[0]) && (Y_2 == center[1]) && (X_3 == left[0]) && (Y_3 == left[1]) && (!aligned_flag)) {
            //console.log(1);
            triangle_1 = new Float32Array([0, 0, right[0], right[1], center[0], center[1]]);
            triangle_2 = new Float32Array([0, 0, shadow1_x, shadow1_y, center[0], center[1]]);
            triangle_3 = new Float32Array([shadow1_x, shadow1_y, center[0], center[1], left[0], left[1]]);
            triangle_4 = new Float32Array([shadow1_x, shadow1_y, shadow2_x, shadow2_y, left[0], left[1]]);
            draw_shadow(gl, webgl, triangle_1, scaled_width, scaled_height, 3);
            draw_shadow(gl, webgl, triangle_2, scaled_width, scaled_height, 3);
            draw_shadow(gl, webgl, triangle_3, scaled_width, scaled_height, 3);
            draw_shadow(gl, webgl, triangle_4, scaled_width, scaled_height, 3);
        }
        else if ((X_1 == left[0]) && (Y_1 == left[1]) && (X_2 == right[0]) && (Y_2 == right[1]) && (X_3 == center[0]) && (Y_3 == center[1]) && (shadow2_pos_0_1 == sun_pos_0_1) && (!aligned_flag)) {
            //console.log(2);
            [X_4, Y_4] = line_intersection([0, 0], [shadow1_x, shadow1_y], [sun_x, sun_y], [shadow2_x, shadow2_y]);
            triangle_1 = new Float32Array([0, 0, left[0], left[1], right[0], right[1]]);
            triangle_2 = new Float32Array([0, 0, shadow1_x, shadow1_y, right[0], right[1]]);
            triangle_3 = new Float32Array([shadow1_x, shadow1_y, shadow2_x, shadow2_y, X_4, Y_4]);
            draw_shadow(gl, webgl, triangle_1, scaled_width, scaled_height, 3);
            draw_shadow(gl, webgl, triangle_2, scaled_width, scaled_height, 3);
            draw_shadow(gl, webgl, triangle_3, scaled_width, scaled_height, 3);
        }
        else if ((X_1 == left[0]) && (Y_1 == left[1]) && (X_2 == right[0]) && (Y_2 == right[1]) && (X_3 == center[0]) && (Y_3 == center[1]) && (shadow2_pos_0_1 != sun_pos_0_1) && (!aligned_flag)) {
            //console.log(3);
            triangle_1 = new Float32Array([0, 0, left[0], left[1], right[0], right[1]]);
            triangle_2 = new Float32Array([0, 0, shadow1_x, shadow1_y, right[0], right[1]]);
            draw_shadow(gl, webgl, triangle_1, scaled_width, scaled_height, 3);
            draw_shadow(gl, webgl, triangle_2, scaled_width, scaled_height, 3);
        }
        else if ((X_1 == right[0]) && (Y_1 == right[1]) && (X_2 == left[0]) && (Y_2 == left[1]) && (X_3 == center[0]) && (Y_3 == center[1]) && (shadow2_pos_0_1 == sun_pos_0_1) && (!aligned_flag)) {
            //console.log(4);
            [X_4, Y_4] = line_intersection([0, 0], [shadow1_x, shadow1_y], [sun_x, sun_y], [shadow2_x, shadow2_y]);
            triangle_1 = new Float32Array([0, 0, right[0], right[1], left[0], left[1]]);
            triangle_2 = new Float32Array([0, 0, shadow1_x, shadow1_y, left[0], left[1]]);
            triangle_3 = new Float32Array([shadow1_x, shadow1_y, shadow2_x, shadow2_y, X_4, Y_4]);
            draw_shadow(gl, webgl, triangle_1, scaled_width, scaled_height, 3);
            draw_shadow(gl, webgl, triangle_2, scaled_width, scaled_height, 3);
            draw_shadow(gl, webgl, triangle_3, scaled_width, scaled_height, 3);
        }
        else if ((X_1 == right[0]) && (Y_1 == right[1]) && (X_2 == left[0]) && (Y_2 == left[1]) && (X_3 == center[0]) && (Y_3 == center[1]) && (shadow2_pos_0_1 != sun_pos_0_1) && (!aligned_flag)) {
            //console.log(5);
            triangle_1 = new Float32Array([0, 0, right[0], right[1], left[0], left[1]]);
            triangle_2 = new Float32Array([0, 0, shadow1_x, shadow1_y, left[0], left[1]]);
            draw_shadow(gl, webgl, triangle_1, scaled_width, scaled_height, 3);
            draw_shadow(gl, webgl, triangle_2, scaled_width, scaled_height, 3);
        }
        else if ((X_1 == center[0]) && (Y_1 == center[1]) && (X_2 == left[0]) && (Y_2 == left[1]) && (X_3 == right[0]) && (Y_3 == right[1]) && (zero_pos_1_2 == sun_pos_1_2) && (!aligned_flag)) {
            //console.log(6);
            [X_4, Y_4] = line_intersection([shadow1_x, shadow1_y], [shadow2_x, shadow2_y], [0, 0], [sun_x, sun_y]);
            triangle_1 = new Float32Array([shadow1_x, shadow1_y, left[0], left[1], right[0], right[1]]);
            triangle_2 = new Float32Array([shadow1_x, shadow1_y, shadow2_x, shadow2_y, right[0], right[1]]);
            triangle_3 = new Float32Array([0, 0, shadow1_x, shadow1_y, X_4, Y_4]);
            draw_shadow(gl, webgl, triangle_1, scaled_width, scaled_height, 3);
            draw_shadow(gl, webgl, triangle_2, scaled_width, scaled_height, 3);
            draw_shadow(gl, webgl, triangle_3, scaled_width, scaled_height, 3);
        }
        else if ((X_1 == center[0]) && (Y_1 == center[1]) && (X_2 == left[0]) && (Y_2 == left[1]) && (X_3 == right[0]) && (Y_3 == right[1]) && (zero_pos_1_2 != sun_pos_1_2) && (!aligned_flag)) {
            //console.log(7);
            triangle_1 = new Float32Array([shadow1_x, shadow1_y, left[0], left[1], right[0], right[1]]);
            triangle_2 = new Float32Array([shadow1_x, shadow1_y, shadow2_x, shadow2_y, right[0], right[1]]);
            draw_shadow(gl, webgl, triangle_1, scaled_width, scaled_height, 3);
            draw_shadow(gl, webgl, triangle_2, scaled_width, scaled_height, 3);
        }
        else if ((X_1 == center[0]) && (Y_1 == center[1]) && (X_2 == right[0]) && (Y_2 == right[1]) && (X_3 == left[0]) && (Y_3 == left[1]) && (zero_pos_1_2 == sun_pos_1_2) && (!aligned_flag)) {
            //console.log(8);
            [X_4, Y_4] = line_intersection([shadow1_x, shadow1_y], [shadow2_x, shadow2_y], [0, 0], [sun_x, sun_y]);
            triangle_1 = new Float32Array([shadow1_x, shadow1_y, right[0], right[1], left[0], left[1]]);
            triangle_2 = new Float32Array([shadow1_x, shadow1_y, shadow2_x, shadow2_y, left[0], left[1]]);
            triangle_3 = new Float32Array([0, 0, shadow1_x, shadow1_y, X_4, Y_4]);
            draw_shadow(gl, webgl, triangle_1, scaled_width, scaled_height, 3);
            draw_shadow(gl, webgl, triangle_2, scaled_width, scaled_height, 3);
            draw_shadow(gl, webgl, triangle_3, scaled_width, scaled_height, 3);
        }
        else if ((X_1 == center[0]) && (Y_1 == center[1]) && (X_2 == right[0]) && (Y_2 == right[1]) && (X_3 == left[0]) && (Y_3 == left[1]) && (zero_pos_1_2 != sun_pos_1_2) && (!aligned_flag)) {
            //console.log(9);
            triangle_1 = new Float32Array([shadow1_x, shadow1_y, right[0], right[1], left[0], left[1]]);
            triangle_2 = new Float32Array([shadow1_x, shadow1_y, shadow2_x, shadow2_y, left[0], left[1]]);
            draw_shadow(gl, webgl, triangle_1, scaled_width, scaled_height, 3);
            draw_shadow(gl, webgl, triangle_2, scaled_width, scaled_height, 3);
        }
        else if(aligned_flag) {
            //console.log(10);
            points_dist = points_distance([sun_x, sun_y], [0, 0], [shadow2_x, shadow2_y], [shadow1_x, shadow1_y]);
            if (points_dist == 2) {triangle_1 = new Float32Array([X_3 - 0.0225, -2.5, X_3 + 0.0225, -2.5, shadow2_x - 0.0225, shadow2_y, shadow2_x + 0.0225, shadow2_y]);}
            else if (points_dist == 1) {triangle_1 = new Float32Array([X_3 - 0.0225, -2.5, X_3 + 0.0225, -2.5, shadow1_x - 0.0225, shadow1_y, shadow1_x + 0.0225, shadow1_y]);}
            else {triangle_1 = new Float32Array([X_1 - 0.0225, -2.5, X_1 + 0.0225, -2.5, -0.0225, 0, +0.0225, 0]);}
            draw_shadow(gl, webgl, triangle_1, scaled_width, scaled_height, 4);
        }
    }
};
// Draw a part of the shadow
function draw_shadow(gl, webgl, shape, scaled_width, scaled_height, num_vertices)
{
    let shadow;
    gl.bindBuffer(gl.ARRAY_BUFFER, webgl.shad_buffer);
    gl.bufferData(gl.ARRAY_BUFFER, shape, gl.STATIC_DRAW);
    shadow = webgl.shadow;
    gl.useProgram(shadow.program);
    gl.uniform2f(shadow.u_aspect, scaled_width, scaled_height);
    gl.bindBuffer(gl.ARRAY_BUFFER, webgl.shad_buffer);
    gl.enableVertexAttribArray(shadow.a_point);
    gl.vertexAttribPointer(shadow.a_point, 2, gl.FLOAT, false, 0, 0);
    gl.uniform2f(shadow.u_attach, 0, 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, num_vertices);
}
// Create a WebGL program using vertex and fragment shaders 
function render_element(gl, vert, frag)
{
    let vertex_shader = gl.createShader(gl.VERTEX_SHADER);
    let fragment_shader = gl.createShader(gl.FRAGMENT_SHADER);
    let program = gl.createProgram();

    gl.shaderSource(vertex_shader, 'precision mediump float;' + vert);
    
    gl.shaderSource(fragment_shader, 'precision mediump float;' + frag);
    gl.compileShader(vertex_shader);
    gl.compileShader(fragment_shader);
    
    gl.attachShader(program, vertex_shader);
    gl.attachShader(program, fragment_shader);
    gl.linkProgram(program);
    
    gl.deleteShader(vertex_shader);
    gl.deleteShader(fragment_shader);
    let result = {program: program};
    let n_attrib = gl.getProgramParameter(program, gl.ACTIVE_ATTRIBUTES);
    for (let a = 0; a < n_attrib; a++)
    {
        let name = gl.getActiveAttrib(program, a).name;
        result[name] = gl.getAttribLocation(program, name);
    }
    let n_uniform = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
    for (let u = 0; u < n_uniform; u++)
    {
        let name = gl.getActiveUniform(program, u).name;
        result[name] = gl.getUniformLocation(program, name);
    }
    return result;
};
// Initialize resources for rendering a scene with a double pendulum
function glRenderer(gl, max_tailLen)
{
    let webgl = {};
    let alpha;
    let v;

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    webgl.quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, webgl.quad);
    gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);
    webgl.tailb = gl.createBuffer();
    webgl.tailpoly = new Float32Array(max_tailLen * 4);
    gl.bindBuffer(gl.ARRAY_BUFFER, webgl.tailb);
    gl.bufferData(gl.ARRAY_BUFFER, webgl.tailpoly.byteLength, gl.STREAM_DRAW);

    webgl.alpha = gl.createBuffer();
    alpha = new Float32Array(max_tailLen * 2);
    for (let i = 0; i < alpha.length; i++) {
        v = (i + 1) / alpha.length;
        alpha[i] = 1 - v;
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, webgl.alpha);
    gl.bufferData(gl.ARRAY_BUFFER, alpha, gl.STATIC_DRAW);

    webgl.light = render_element(gl, sunShader.vert, sunShader.frag);
    webgl.pivot = render_element(gl, pivotShader.vert, pivotShader.frag);
    webgl.mass = render_element(gl, massShader.vert, massShader.frag);
    webgl.bar  = render_element(gl, barShader.vert, barShader.frag);
    webgl.tail = render_element(gl, tailShader.vert, tailShader.frag);

    let shape = new Float32Array([0, 0, 1, 1, 0, 1]);
    webgl.shad_buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, webgl.shad_buffer);
    gl.bufferData(gl.ARRAY_BUFFER, shape, gl.STATIC_DRAW);
    webgl.shadow = render_element(gl, shadShader.vert, shadShader.frag);

    webgl.renderScene = function(pendulum, tail, m1_tail, sun_pos, shadow) {render_pend(gl, webgl, pendulum, tail, m1_tail, sun_pos, shadow);};
    return webgl;
}

(function()
{
    const tail_Color = [255/255, 0/255, 0/255];
    let double_pend = new double_pendulum({TailColor:tail_Color, initial_conditions:[0.0, 0.0, 0.0, 0.0]});
    let c2d = document.getElementById('c2d');
    let c3d = document.getElementById('c3d');
    let canvas = c3d;
    let ww = window.innerWidth;
    let wh = window.innerHeight;
    let running = true;
    let gl = c3d.getContext('webgl')
    let renderer = new glRenderer(gl, tailLenghh_Max);
    let tail = false;
    let m1_tail = false;
    let pend_color;
    let sun_pos = [0, 1];
    let [light, shadow] = [false, false];
    let last_tail = tail;
    let damping_coeff;
    let solver_method;
    const rgb_tail_Color = [255, 0, 0];
    let movement_1 = false;
    let movement_2 = false;
    let new_tail_Color;
    let ang1, ang2;
    let last_frame = 0.0;

    if (canvas.width != ww || canvas.height != wh)
    {
        canvas.width = ww;
        canvas.height = wh; 
    }

    // Add references to control panel buttons 
    let PauseResume_Btn = document.querySelector(".pause-resume");
    let OnOffColor_Btn = document.querySelector(".on-off_color");
    let Restart_Btn = document.querySelector(".restart");
    let RandomRestart_Btn = document.querySelector(".random_restart");
    let DayNight = document.querySelector(".on-off_light");
    let RedInput = document.getElementById('red');
    const RedVal = document.getElementById('red-value');
    let GreenInput = document.getElementById('green');
    const GreenVal = document.getElementById('green-value');
    let BlueInput = document.getElementById('blue');
    const BlueVal = document.getElementById('blue-value');
    let Perpetual = document.getElementById('perpetual_button');
    let Mass1_Tail = document.getElementById('mass1_tail_button');
    let controlsDiv = document.getElementById("controls-div");
    let diffEqSolverSelect = controlsDiv.querySelector("select");
    let SunPosInput = document.getElementById('sun');

    c2d.style.display = 'none';
    damping_coeff = 0.0;
    solver_method = 0;
    [RedInput.value, GreenInput.value, BlueInput.value] = rgb_tail_Color;
    [RedVal.textContent, GreenVal.textContent, BlueVal.textContent] = rgb_tail_Color;

    /* Controls */
    // Pause/Resume the simulation
    PauseResume_Btn.addEventListener("click", function () {
        if (running) {
            PauseResume_Btn.textContent = 'Resume';
            running = !running;}
        else {
            PauseResume_Btn.textContent = 'Pause';
            running = !running;
        }
    });
    // Turn on/off the tail
    OnOffColor_Btn.addEventListener("click", function () {
        if (tail) {
            OnOffColor_Btn.textContent = 'Tails Off';
            double_pend.m2_tail.length = 0;
        }
        else {
            OnOffColor_Btn.textContent = 'Tails On';
        }
        tail = !tail;
    });
    // Reset the double_pendulum from a rest position
    Restart_Btn.addEventListener("click", function () {
        pend_color = double_pend.DoublePendulumColor;
        double_pend = new double_pendulum({initial_conditions:[0.0, 0.0, 0.0, 0.0], TailColor:tail_Color, DoublePendulumColor:pend_color});
        [RedInput.value, GreenInput.value, BlueInput.value] = rgb_tail_Color;
        [RedVal.textContent, GreenVal.textContent, BlueVal.textContent] = rgb_tail_Color;
    });
    // Reset the double_pendulum from a random position
    RandomRestart_Btn.addEventListener("click", function () {
        pend_color = double_pend.DoublePendulumColor;
        double_pend = new double_pendulum({TailColor:tail_Color, DoublePendulumColor:pend_color});
        [RedInput.value, GreenInput.value, BlueInput.value] = rgb_tail_Color;
        [RedVal.textContent, GreenVal.textContent, BlueVal.textContent] = rgb_tail_Color;
    });
    //Switch Day and Night
    DayNight.addEventListener("click", function () {
        if (light) {
            DayNight.textContent = 'Day';
            const html = document.querySelector('html');
            html.style.backgroundColor = `rgb(${129}, ${157}, ${248})`;
            const body = document.querySelector('body');
            body.style.backgroundColor = `rgb(${129}, ${157}, ${248})`;
            double_pend.DoublePendulumColor = [0/255, 0/255, 0/255];
            const bar = document.querySelector('.sunPosition');
            bar.style.accentColor = `rgb(${245}, ${245}, ${245})`;
        }
        else {
            DayNight.textContent = 'Night';
            const html = document.querySelector('html');
            html.style.backgroundColor = `rgb(${248}, ${246}, ${129})`;
            const body = document.querySelector('body');
            body.style.backgroundColor = `rgb(${248}, ${246}, ${129})`;
            double_pend.DoublePendulumColor = [80/255, 80/255, 80/255];
            const bar = document.querySelector('.sunPosition');
            bar.style.accentColor = `rgb(${255}, ${215}, ${0})`;
        }
        [light, shadow] = [!light, !shadow];
    });
    // Control the red value [0, 255] of the tail
    RedInput.addEventListener("click", function () {
        double_pend.TailColor = [(RedInput.value / 255), double_pend.TailColor[1], double_pend.TailColor[2]];
        RedInput.addEventListener("input", function () {RedVal.textContent = RedInput.value;});
    });
    // Control the green value [0, 255] of the tail
    GreenInput.addEventListener("click", function () {
        double_pend.TailColor = [double_pend.TailColor[0], (GreenInput.value / 255), double_pend.TailColor[2]];
        GreenInput.addEventListener("input", function () {GreenVal.textContent = GreenInput.value;});
    });
    // Control the blue value [0, 255] of the tail
    BlueInput.addEventListener("click", function () {
        double_pend.TailColor = [double_pend.TailColor[0], double_pend.TailColor[1], (BlueInput.value / 255)];
        BlueInput.addEventListener("input", function () {BlueVal.textContent = BlueInput.value;});
    });
    // Turn on/off the perpetual flag
    Perpetual.addEventListener("change", function () {
        if (Perpetual.checked) {damping_coeff = 0.05;}
        else {damping_coeff = 0.0;}
    });
    Mass1_Tail.addEventListener("change", function () {
        if (Mass1_Tail.checked) {
            m1_tail = true;
            double_pend.m1_tail.length = 0;
            }
        else {
            m1_tail = false;
        }
    });
    // Choose the method for solving ordinary differential equations (ODE)
    diffEqSolverSelect.addEventListener("change", function() {
        const selectedSolver = diffEqSolverSelect.value;
        if (selectedSolver == "Runge-Kutta Method") {solver_method = 0;}
        else if (selectedSolver == "Eulers Method") {solver_method = 1;}
        else if (selectedSolver == "Adams-Bashfort Method") {solver_method = 2;}
    });
    // Select one of the two masses
    window.addEventListener("click", function(event) {
        if (!running) {
            last_tail = tail;
            const x = event.clientX;
            const y = event.clientY;
            const [x1, y1, x2, y2] = double_pend.positions();
            let smallest_dim = Math.min(canvas.width, canvas.height);
            let m_x = ((2 / ((2*barLength)/(canvas.width/smallest_dim))) / canvas.width);
            let q_x = (m_x * canvas.width) / 2;
            let m_y = ((2 / ((2*barLength)/(canvas.height/smallest_dim))) / canvas.height);
            let q_y = (m_y * canvas.height)/ 2;
            const x_prime = m_x*x - q_x;
            const y_prime = m_y*y - q_y;
            if ((Math.abs(x1) - Math.abs(x_prime)<=0.05) && (Math.abs(x1) - Math.abs(x_prime)>=-0.05) && (Math.abs(y1) - Math.abs(y_prime)<=0.05) && (Math.abs(y1) - Math.abs(y_prime)>=-0.05)) {
                tail = true;
                movement_1 = true;
                running = true;
                [ang1, ang2] = double_pend.angles();
            }
            else if ((Math.abs(x2) - Math.abs(x_prime)<=0.05) && (Math.abs(x2) - Math.abs(x_prime)>=-0.05) && (Math.abs(y2) - Math.abs(y_prime)<=0.05) && (Math.abs(y2) - Math.abs(y_prime)>=-0.05)) {
                tail = true;
                movement_2 = true;
                running = true;
                [ang1, ang2] = double_pend.angles();
            }}
        else {
            if (movement_1 || movement_2) {
                [ang1, ang2] = double_pend.angles();
                new_tail_Color = [double_pend.TailColor[0], double_pend.TailColor[1], double_pend.TailColor[2]];
                pend_color = double_pend.DoublePendulumColor;
                double_pend = new double_pendulum({initial_conditions:[ang1, ang2, 0, 0], TailColor:new_tail_Color, DoublePendulumColor:pend_color});
                movement_1 = false;
                movement_2 = false;
                //running = true;
                running = false;
                tail = last_tail;
                //PauseResume_Btn.textContent = 'Pause';
            } 
        }});
    // Move the selected mass
    window.addEventListener("mousemove", function(event) {
        let mouse_x, mouse_y;
        if (movement_1) {
            mouse_x = event.clientX;
            mouse_y = event.clientY;
            const x_prime = (9.5 / 1536)*mouse_x - 4.75;
            const y_prime = (4.36 / 703)*mouse_y - 2.18;
            ang1 = Math.atan2(x_prime, y_prime);
        }
        else if (movement_2) {
            mouse_x = event.clientX;
            mouse_y = event.clientY;
            const y_prime = (4.36 / 703)*mouse_y - 2.18;
            const x_prime = (9.5 / 1536)*mouse_x - 4.75;
            ang2 = Math.atan2(x_prime, y_prime);
        }});
    // Control the position of the sun
    SunPosInput.addEventListener("click", function () {
        sun_pos = [SunPosInput.value, 1];
    });
    
    function animation(current_frame)
    {
        let dt = Math.min(current_frame - last_frame, dt_Max);

        if (running) {
            if (movement_1) {double_pend.manual_step_1(ang1);}
            else if (movement_2) {double_pend.manual_step_2(ang2);}
            else {double_pend.step(dt/1000, solver_method, damping_coeff);}
        }
        renderer.renderScene(double_pend, tail, m1_tail, sun_pos, shadow);
        last_frame = current_frame;
        window.requestAnimationFrame(animation);
    }
    window.requestAnimationFrame(animation);
}());