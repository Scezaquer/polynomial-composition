from manim import *  # or: from manimlib import *

from manim_slides import Slide
from numpy import polynomial as P
from main import gradient_descent
from polynomial_utils import l_inf_norm, compose_layers
from carleman_approach import carleman_solver


class BasicExample(Slide):
    def construct(self):
        # Title slide
        title = Text("MATH 578", font_size=100)
        self.play(Write(title))
        subtitle = Text(
            "Fall 2024, Aurélien Bück-Kaeffer, 261032581", font_size=30)
        subtitle.next_to(subtitle, DOWN, buff=0.5)
        self.play(FadeIn(subtitle))

        self.play(title.animate.shift(UP*3), subtitle.animate.shift(UP*3))

        definition_and_composition = MathTex(
            r"\forall x \in \mathbb{R}, p_i(x) & = \sum_{k=0}^{n_i} c_{i,k} x^k \\"
            r"(p_1 \circ p_2 \circ p_3)(x) & = p_1(p_2(p_3(x)))"
        )
        definition_and_composition.arrange(DOWN, aligned_edge=LEFT)
        self.play(Write(definition_and_composition))

        self.next_slide()

        # Problem statement

        text_to_color_map = {
            "p_1": YELLOW,
            "p_2": YELLOW,
            "p_3": YELLOW,
            "p_n": YELLOW,
            "p_i": YELLOW,
            "p_k": YELLOW,
            "b_0": RED,
            "b_i": RED,
            "b_m": RED,
            "c_{i,k}": RED,
            "Q": BLUE
        }

        problem_statement = Text("Introduction", font_size=50)
        # Transform the previous slide into the next one
        self.play(FadeOut(title), FadeOut(subtitle), ReplacementTransform(
            definition_and_composition, problem_statement))
        self.play(problem_statement.animate.shift(UP*3))

        given_polynomials = MathTex(
            r"\text{Given polynomials }", r"p_1, p_2, \ldots, p_n", tex_to_color_map=text_to_color_map)
        composition_result = MathTex(
            r"Q(x) = (", r"p_1 \circ p_2 \circ \ldots \circ p_n", r")(x)", tex_to_color_map=text_to_color_map)
        q_as_a_poly = MathTex(
            r"Q(x) = \sum_{k=0}^{m} b_k x^k", tex_to_color_map=text_to_color_map)
        degree_bound = MathTex(
            r"m = \prod_{i=1}^n \deg(", r"p_i", r")", tex_to_color_map=text_to_color_map)

        given_polynomials.next_to(problem_statement, DOWN, buff=0.5)
        composition_result.next_to(given_polynomials, DOWN, buff=0.3)
        q_as_a_poly.next_to(composition_result, DOWN, buff=0.3)
        degree_bound.next_to(q_as_a_poly, DOWN, buff=0.3)

        self.play(Write(given_polynomials))
        self.play(Write(composition_result))

        self.next_slide()
        self.play(TransformFromCopy(composition_result, q_as_a_poly))
        self.play(Write(degree_bound))

        self.next_slide()
        self.play(FadeOut(problem_statement), FadeOut(given_polynomials), FadeOut(
            composition_result), FadeOut(degree_bound), FadeOut(q_as_a_poly))

        title = Text("Problem Statement", font_size=50)
        self.play(Write(title))
        self.play(title.animate.shift(UP*3))

        condition = MathTex(r"\text{ Provided } Q(x)",
                            tex_to_color_map=text_to_color_map)
        minimize_error = MathTex(
            r"\text{Find } p_1, p_2, \ldots, p_n \text{ such that } Q(x) \approx (p_1 \circ p_2 \circ \ldots \circ p_n)(x)", tex_to_color_map=text_to_color_map)
        condition.next_to(title, DOWN, buff=0.5)
        minimize_error.next_to(condition, DOWN, buff=0.5)

        self.play(Write(condition))
        self.play(Write(minimize_error))

        self.next_slide()

        reformulated_condition = MathTex(
            r"\text{ Provided the coefficients } b_0, ..., b_m \text{ such that } Q(x) = \sum_{i=0}^{m} b_ix^i", tex_to_color_map=text_to_color_map)
        reformulated_minimize_error1 = MathTex(
            r"\text{Find the coefficients } c_{i,k} \text{ of } p_k := \sum_{i=0}^{n_k}c_{i,k}x^i \text{ such that }", tex_to_color_map=text_to_color_map)
        reformulated_minimize_error2 = MathTex(
            r"Q(x) \approx (p_1 \circ p_2 \circ \ldots \circ p_n)(x)", tex_to_color_map=text_to_color_map)

        reformulated_condition.next_to(title, DOWN, buff=0.5)
        reformulated_minimize_error1.next_to(
            reformulated_condition, DOWN, buff=0.5)
        reformulated_minimize_error2.next_to(
            reformulated_minimize_error1, DOWN, buff=0.5)

        self.play(ReplacementTransform(condition, reformulated_condition))
        self.play(ReplacementTransform(minimize_error, reformulated_minimize_error1), Write(
            reformulated_minimize_error2))

        self.next_slide()
        minimize_some_norm = MathTex(
            r"\| Q - (p_1 \circ p_2 \circ \ldots \circ p_n)\| \text{ is minimized}", tex_to_color_map=text_to_color_map)
        minimize_some_norm.next_to(
            reformulated_minimize_error1, DOWN, buff=0.5)
        self.play(ReplacementTransform(
            reformulated_minimize_error2, minimize_some_norm))

        self.next_slide()
        self.play(FadeOut(title), FadeOut(reformulated_condition), FadeOut(
            reformulated_minimize_error1), FadeOut(minimize_some_norm))

        # Comparison with Machine Learning

        text = Text("This looks like Machine Learning!", font_size=50)
        self.play(Write(text))
        self.play(text.animate.shift(UP*3))

        ml = MathTex(
            r"\text{ML minimizes } \| \vec{Y} - \sigma(A\sigma(B\sigma(\ldots C\vec{X}))) \|")
        ml.next_to(text, DOWN, buff=0.5)
        self.play(Write(ml))

        polycomp = MathTex(
            r"\text{We minimize } \| Q - p_1(p_2(\ldots(p_n(x))) \|",)
        polycomp.next_to(ml, DOWN, buff=0.5)
        self.play(Write(polycomp))

        self.next_slide()

        main_difference1 = Text(
            "The main difference is that ML optimizes", font_size=30)
        main_difference2 = MathTex(
            r"\text{matrices } A, B, \ldots, C", r"\text{ instead of polynomial coefficients } c_{i,k}")

        main_difference1.next_to(polycomp, DOWN, buff=0.5)
        main_difference2.next_to(main_difference1, DOWN, buff=0.5)
        self.play(Write(main_difference1))
        self.play(Write(main_difference2))

        self.next_slide()
        self.play(FadeOut(text), FadeOut(ml), FadeOut(polycomp),
                  FadeOut(main_difference1), FadeOut(main_difference2))

        # Gradient Descent slide

        first_approach = Text("First Approach", font_size=50)
        self.play(Write(first_approach))
        title = Text("Gradient Descent", font_size=50)
        self.play(ReplacementTransform(first_approach, title))
        self.play(title.animate.shift(UP*3))

        gradient_descent_description = Text(
            "An optimization algorithm to minimize a function by\niteratively moving towards the steepest descent direction.",
            font_size=30
        )
        gradient_descent_description.next_to(title, DOWN, buff=0.5)
        self.play(Write(gradient_descent_description))

        gradient_descent_formula = MathTex(
            r"\theta_{t+1} := \theta_t - \gamma \nabla_\theta \mathcal{L}(\theta)",
            tex_to_color_map={"\\theta": RED, "L": YELLOW, "\\nabla": GREEN}
        )
        gradient_descent_formula.next_to(
            gradient_descent_description, DOWN, buff=0.5)
        self.play(Write(gradient_descent_formula))

        self.next_slide()

        self.play(FadeOut(gradient_descent_description), FadeOut(
            title), gradient_descent_formula.animate.shift(UP*2.5))

        # Gradient Descent on a Parabola
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-1, 9, 1],
            axis_config={"color": BLUE}
        )
        parabola = axes.plot(lambda x: x**2, color=YELLOW)
        self.play(Create(axes), Create(parabola))

        x, y = 2, 4
        point = Dot(axes.coords_to_point(x, y), color=RED)
        self.play(FadeIn(point))

        for _ in range(4):
            slope = 2 * x  # Derivative of x^2 is 2x
            intercept = y - slope * x
            tangent = axes.plot(lambda t: slope * t + intercept, color=GREEN)
            self.play(Create(tangent))

            new_x = x - 0.1 * slope  # Gradient descent step
            new_y = new_x**2
            new_point = Dot(axes.coords_to_point(new_x, new_y), color=RED)
            self.play(Transform(point, new_point))
            self.play(FadeOut(tangent))
            x, y = new_x, new_y

        self.next_slide()

        # Apply partial transparency to the graph
        self.play(axes.animate.fade(0.8), parabola.animate.fade(
            0.8), point.animate.fade(0.8))

        loss_function = MathTex(
            r"\mathcal{L}(\theta) := \| Q - (p_1 \circ p_2 \circ \ldots \circ p_n)\|",
            tex_to_color_map={"\\theta": RED, "L": YELLOW}
        )
        loss_function.move_to(ORIGIN)
        self.play(TransformFromCopy(
            gradient_descent_formula[-4], loss_function))

        parameters = MathTex(
            r"\theta = (c_{i,k})_{i,k}",
            tex_to_color_map={"\\theta": RED}
        )

        parameters.next_to(loss_function, DOWN, buff=0.5)
        self.play(TransformFromCopy(loss_function[3], parameters))

        self.next_slide()

        self.play(loss_function.animate.shift(UP*2),
                  parameters.animate.shift(UP*2))

        gradient = MathTex(
            r"\text{We need to compute }\nabla_\theta \mathcal{L}(\theta)",
            tex_to_color_map={"\\theta": RED, "\\nabla": GREEN, "L": YELLOW}
        )

        self.play(Write(gradient))

        self.next_slide()

        self.play(FadeOut(gradient_descent_formula), FadeOut(loss_function), FadeOut(parameters),
                  gradient.animate.shift(UP*3), FadeOut(axes), FadeOut(parabola), FadeOut(point))

        # Gradient computation
        gradient_computation = MathTex(
            r"\nabla_\theta \mathcal{L}(\theta) = ( \frac{\partial \mathcal{L}}{\partial c_{i,k}} )_{i,k}",
            tex_to_color_map={"\\theta": RED, "\\nabla": GREEN}
        )
        gradient_computation.next_to(gradient, DOWN, buff=0.5)
        self.play(Write(gradient_computation))

        self.next_slide()

        partial_derivative_lhs = MathTex(r"\frac{\partial \mathcal{L}}{\partial c_{i,k}} = ")
        partial_derivative_part1 = MathTex(r"\frac{\partial \mathcal{L}}{\partial P_1}")
        partial_derivative_part2 = MathTex(r"\frac{\partial P_1}{\partial P_2} \ldots \frac{\partial P_{k-1}}{\partial P_k}")
        partial_derivative_part3 = MathTex(r"\frac{\partial P_k}{\partial c_{i,k}}")

        partial_derivative_group = VGroup(
            partial_derivative_lhs, partial_derivative_part1, partial_derivative_part2, partial_derivative_part3)

        partial_derivative_group.next_to(gradient_computation, DOWN, buff=0.5)
        partial_derivative_group.arrange(buff=0.2)
        # partial_derivative_part2_.next_to(gradient_computation, DOWN, buff=0.5)
        # partial_derivative_part1_.next_to(partial_derivative_part2_, LEFT, buff=0.1)
        # partial_derivative_lhs_.next_to(partial_derivative_part1_, LEFT, buff=0.2)
        # partial_derivative_part3_.next_to(partial_derivative_part2_, RIGHT, buff=0.1)

        # partial_derivative.next_to(gradient_computation, DOWN, buff=0.5)
        # self.play(Write(partial_derivative_lhs_, run_time=0.4))
        # self.play(Write(partial_derivative_part1_, run_time=0.4))
        # self.play(Write(partial_derivative_part2_, run_time=0.4))
        # self.play(Write(partial_derivative_part3_, run_time=0.4))
        self.play(Write(partial_derivative_group))

        self.next_slide()
        self.play(FadeOut(gradient), FadeOut(gradient_computation))

        # partial_derivative_lhs = MathTex(r"\frac{\partial \mathcal{L}}{\partial a_{i,k}} = ")
        # partial_derivative_part1 = MathTex(r"\frac{\partial \mathcal{L}}{\partial p_1}")
        # partial_derivative_part2 = MathTex(r"\frac{\partial p_1}{\partial p_2} \ldots \frac{\partial p_{k-1}}{\partial p_k}")
        # partial_derivative_part3 = MathTex(r"\frac{\partial p_k}{\partial a_{i,k}}")

        # partial_derivative_part2.scale(0.5).shift(UP*3)
        # partial_derivative_part1.scale(0.5).next_to(partial_derivative_part2, LEFT, buff=0.05)
        # partial_derivative_part3.scale(0.5).next_to(partial_derivative_part2, buff=0.05)
        # partial_derivative_lhs.scale(0.5).next_to(partial_derivative_part1, LEFT, buff=0.1)

        # self.play(ReplacementTransform(partial_derivative_part1_, partial_derivative_part1),
        #           ReplacementTransform(partial_derivative_part2_, partial_derivative_part2),
        #           ReplacementTransform(partial_derivative_part3_, partial_derivative_part3),
        #           ReplacementTransform(partial_derivative_lhs_, partial_derivative_lhs))

        self.play(partial_derivative_group.animate.scale(0.5).shift(UP*3))

        activations_def = MathTex(
            r"\text{We define the activations } \alpha_0 \ldots \alpha_n \text{ as }",
            tex_to_color_map={"\\alpha": YELLOW}
        )
        alpha = MathTex(r"\alpha_n &= x\\"
                        r"\alpha_{n-1} &= p_n(\alpha_0)\\"
                        r"\alpha_{n-2} &= p_{n-1}(\alpha_{n-1})\\"
                        r"\ldots\\"
                        r"\alpha_0 &= p_1(\alpha_1)",
                        tex_to_color_map={"\\alpha": YELLOW})

        activations_def.shift(UP*2)
        alpha.next_to(activations_def, DOWN, buff=0.3)

        self.play(Write(activations_def))
        self.play(Write(alpha, run_time=3))

        self.next_slide()
        self.play(FadeOut(activations_def), FadeOut(alpha))

        part3 = MathTex(r"\frac{\partial P_k}{\partial c_{i,k}}")
        poly = MathTex(
            r"P_k(x) := c_{0,k} + c_{1,k}\alpha_{k+1} + \ldots + c_{n_k,k}(\alpha_{k+1})^{n_k}", tex_to_color_map={"\\alpha": YELLOW})
        equals = MathTex(r" = ")
        x_i = MathTex(r"(\alpha_{k+1})^{i}",
                      tex_to_color_map={"\\alpha": YELLOW})

        part3.next_to(ORIGIN, UP)
        poly.next_to(part3, DOWN)
        # x_i.scale(0.5).next_to(partial_derivative_part2, buff=0.05)

        self.play(ReplacementTransform(partial_derivative_part3, part3))
        self.play(Write(poly))
        self.play(part3.animate.shift(LEFT))

        equals.next_to(part3, RIGHT, buff=0.2)
        x_i.next_to(equals, RIGHT, buff=0.2)

        self.play(Write(VGroup(equals, x_i)))

        self.next_slide()
        self.play(FadeOut(poly), FadeOut(equals), FadeOut(part3), x_i.animate.scale(
            0.5).next_to(partial_derivative_part2, buff=0.05))

        part2 = MathTex(r"\frac{\partial P_{k-1}}{\partial P_k}")
        part2_2 = MathTex(r"\frac{\partial p_{k-1}(p_k(p_{k+1}(\ldots p_n(x))))}{\partial p_k(p_{k+1}(\ldots p_n(x)))")
        part2_3 = MathTex(r"\frac{\partial p_{k-1}(\alpha_k)}{\partial \alpha_k}")
        equals = MathTex(r" = ")
        p_k1 = MathTex(r"p_{k-1}'(\alpha_k)",
                       tex_to_color_map={"\\alpha": YELLOW})
        p1_pk1 = MathTex(
            r"p_1'(\alpha_2) \ldots p_{k-1}'(\alpha_k)", tex_to_color_map={"\\alpha": YELLOW})

        part2.next_to(ORIGIN, UP)
        part2_2.next_to(ORIGIN, UP)
        part2_3.next_to(ORIGIN, UP)
        p1_pk1.scale(0.5).next_to(partial_derivative_part1, buff=0.05)

        self.play(ReplacementTransform(partial_derivative_part2, part2))

        self.next_slide()
        self.play(ReplacementTransform(part2, part2_2))

        self.next_slide()
        self.play(ReplacementTransform(part2_2, part2_3))

        self.next_slide()

        self.play(part2_3.animate.shift(LEFT))
        equals.next_to(part2_3, RIGHT, buff=0.2)
        p_k1.next_to(equals, RIGHT, buff=0.2)

        self.play(Write(VGroup(equals, p_k1)))

        self.next_slide()
        self.play(FadeOut(equals), FadeOut(part2_3), ReplacementTransform(
            p_k1, p1_pk1), x_i.animate.next_to(p1_pk1, buff=0.05))

        part1 = MathTex(r"\frac{\partial \mathcal{L}}{\partial P_1}")
        text = MathTex(
            r"\text{This depends on } \mathcal{L} \text{, so we must pick a metric}")

        part1.next_to(ORIGIN, UP)
        text.next_to(part1, DOWN)

        self.play(ReplacementTransform(partial_derivative_part1, part1))
        self.play(Write(text))
        self.next_slide()

        def_P = MathTex(r"\text{With } P_1(x) := p_1(p_2(\ldots(p_n(x)))")
        def_L = MathTex(
            r"\text{We define } \mathcal{L} := (P_1(x) - Q(x))^2 \forall x \in \mathbb{R}")

        def_P.next_to(part1, DOWN)
        def_L.next_to(def_P, DOWN)

        self.play(FadeOut(text))
        self.play(Write(def_P))
        self.play(Write(def_L))

        self.next_slide()

        part1_derivation_1 = MathTex(r" = ")
        part1_derivation_2 = MathTex(r"2(")
        part1_derivation_3 = MathTex(r"P_1(x) - Q(x)")
        part1_derivation_4 = MathTex(r")")

        self.play(part1.animate.shift(LEFT*2))

        part1_derivation_1.next_to(part1, RIGHT, buff=0.2)
        part1_derivation_2.next_to(part1_derivation_1, RIGHT, buff=0.2)
        part1_derivation_3.next_to(part1_derivation_2, RIGHT, buff=0.05)
        part1_derivation_4.next_to(part1_derivation_3, RIGHT, buff=0.05)

        self.play(Write(VGroup(part1_derivation_1, part1_derivation_2, part1_derivation_3, part1_derivation_4)))

        self.next_slide()

        self.play(FadeOut(part1_derivation_2), FadeOut(part1_derivation_4),
                  part1_derivation_3.animate.next_to(part1_derivation_1, RIGHT, buff=0.2))

        self.next_slide()

        part3_final = MathTex(r"(P_1(x) - Q(x))")
        part3_final.scale(0.5).next_to(p1_pk1, LEFT, buff=0.05)
        self.play(FadeOut(def_P), FadeOut(def_L), FadeOut(part1_derivation_1), FadeOut(part1),
                  ReplacementTransform(part1_derivation_3, part3_final),
                  partial_derivative_lhs.animate.next_to(part3_final, LEFT, buff=0.2))

        partial_derivative_group = VGroup(partial_derivative_lhs, part3_final, p1_pk1, x_i)
        self.play(partial_derivative_group.animate.shift(DOWN*3).scale(2))

        self.next_slide()
        self.play(FadeOut(partial_derivative_group))

        title = Text("Applying all this", font_size=30)

        self.play(Write(title))
        self.play(FadeOut(title))

        p1 = P.Polynomial([-0.33753906, 0.14620071, 0.37282418, - 0.11214395])
        p2 = P.Polynomial([0.1280617, 0.24731691, 0.35288478, 0.40186104])
        p3 = P.Polynomial([0.15562597, 0.40363781, - 0.11945435, 0.02205813])
        target = P.Polynomial([0.14298828, - 1.32111465, 1.14062129, 1.039349, 0.5955707,
                               -1.24627749, 0.01084536, 0.88981526, - 1.27592646,
                               2.01573881, - 0.40165688, 0.66728787, - 0.54458902,
                               -1.63101889, 0.34588164, - 1.27556028, 0.16574406,
                               2.13309033, - 1.86773342, - 1.57719288, - 1.4696854,
                               1.93512148, 1.63620253, - 1.35902556, 0.66833177,
                               1.80089458, 0.14650389, 1.40650117])

        p1_mob = MathTex(f"p_1(x) = {p1.__str__()}")
        p2_mob = MathTex(f"p_2(x) = {p2.__str__()}")
        p3_mob = MathTex(f"p_3(x) = {p3.__str__()}")

        polys_mob = VGroup(p1_mob, p2_mob, p3_mob)
        polys_mob.arrange(DOWN, aligned_edge=LEFT).scale(0.5).shift(UP*3)

        def update1(mob):
            mob.become(MathTex(f"p_1(x) = {p1.__str__()}"))

        def update2(mob):
            mob.become(MathTex(f"p_2(x) = {p2.__str__()}"))

        def update3(mob):
            mob.become(MathTex(f"p_3(x) = {p3.__str__()}"))

        # p1_mob.add_updater(update1)
        # p2_mob.add_updater(update2)
        # p3_mob.add_updater(update3)

        iteration = Text(f"Iteration ")
        iteration_nbr = Text("0")
        iter_grp = VGroup(iteration, iteration_nbr)
        iter_grp.scale(0.5)
        iteration_nbr.next_to(iteration, RIGHT, buff=0.1)
        l_inf_loss = MathTex(r"\|Q(x) - p_1(p_2(p_3(x)))\|_{L^\infty} = " + f"{l_inf_norm(target, compose_layers([p1, p2, p3])):.4f}", tex_to_color_map={"Q": BLUE, "p_1": YELLOW, "p_2": YELLOW, "p_3": YELLOW})

        # Create axes
        axes = Axes(
            x_range=[-0.1, 1.1],
            y_range=[-0.25, 2.5],
            axis_config={"color": BLUE}
        )

        # Create the target polynomial plot
        target_plot = axes.plot(lambda x: target(x), color=BLUE, use_smoothing=False, x_range=[0, 1, 0.001])

        # Create the composed polynomial plot
        composed_plot = axes.plot(lambda x: p3(p2(p1(x))), color=YELLOW, use_smoothing=False, x_range=[0, 1, 0.001])

        # Add the plots to the scene
        self.play(Create(axes), Create(target_plot), Create(composed_plot))

        # Position the polynomials and vs text
        # p1_mob.scale(0.5).to_corner(UL)
        # p2_mob.scale(0.5).next_to(p1_mob, DOWN, buff=0.1)
        # p3_mob.scale(0.5).next_to(p2_mob, DOWN, buff=0.1)
        l_inf_loss.next_to(axes, DOWN, buff=0.5)
        iter_grp.next_to(l_inf_loss, UP, buff=0.25)

        # Add the polynomials and vs text to the scene
        self.play(Write(polys_mob), Write(l_inf_loss), Write(iter_grp))

        self.next_slide()

        for i in range(1, 6):
            polys, _ = gradient_descent(seed=473753, max_iter=5000*i, verbose=False, plot=False)
            #polys, _ = gradient_descent(target, [p1, p2, p3], max_iter=2500*iter, verbose=False, plot=False)

            # Update the polynomials with the result from gradient descent
            p1_, p2_, p3_ = polys

            # Update the composed polynomial plot
            composed = compose_layers(polys)
            new_composed_plot = axes.plot(lambda x: composed(x), color=YELLOW, use_smoothing=False, x_range=[0, 1, 0.001])

            # Update the displayed polynomials and loss
            new_p1_mob = MathTex(f"p_1(x) = {p3_.__str__()}")
            new_p2_mob = MathTex(f"p_2(x) = {p2_.__str__()}")
            new_p3_mob = MathTex(f"p_3(x) = {p1_.__str__()}")
            new_poly_group = VGroup(new_p1_mob, new_p2_mob, new_p3_mob)
            new_poly_group.arrange(DOWN, aligned_edge=LEFT).scale(0.5).shift(UP*3)

            new_iteration_nbr = Text(f"{i*5000}")
            new_iteration_nbr.scale(0.5)
            new_iteration_nbr.next_to(iteration, RIGHT, buff=0.1)
            new_l_inf_loss = MathTex(r"\|Q(x) - p_1(p_2(p_3(x)))\|_{L^\infty} = " + f"{l_inf_norm(target, compose_layers(polys)):.4f}", tex_to_color_map={"Q": BLUE, "p_1": YELLOW, "p_2": YELLOW, "p_3": YELLOW})
            new_l_inf_loss.next_to(axes, DOWN, buff=0.5)

            self.play(ReplacementTransform(iteration_nbr, new_iteration_nbr), ReplacementTransform(composed_plot, new_composed_plot), ReplacementTransform(l_inf_loss, new_l_inf_loss), ReplacementTransform(polys_mob, new_poly_group))
            polys_mob = new_poly_group
            l_inf_loss = new_l_inf_loss
            composed_plot = new_composed_plot
            iteration_nbr = new_iteration_nbr

        self.next_slide()
        self.play(FadeOut(polys_mob), FadeOut(iter_grp), FadeOut(l_inf_loss), FadeOut(composed_plot), FadeOut(axes), FadeOut(target_plot))

        title = Text("Possible improvements", font_size=30)
        title.shift(UP*3)
        self.play(Write(title))

        self.next_slide()
        adam = Text("Using the Adam optimizer", font_size=20)
        adam.shift(UP*2)
        self.play(Write(adam))

        self.next_slide()
        genetic = Text("Picking a better initialization with a genetic algorithm", font_size=20)
        genetic.shift(UP)
        self.play(Write(genetic))

        self.next_slide()
        self.play(FadeOut(adam), FadeOut(genetic))
        
        genetic_alg_comparison_plot = ImageMobject("genetic_alg_comparison_plot.png")

        # "gd": {
        #     "mean": 9.468959728494694,
        #     "median": 0.5242933978823618,
        #     "std": 45.65352169964436
        # },
        # "ga_gd": {
        #     "mean": 0.10964880242599596,
        #     "median": 0.047236476504887825,
        #     "std": 0.17840857566044388
        # }
        
        gd_stats = MathTex(r"\text{Mean: } &9.47\\"
                           r"\text{Median: } &0.52\\"
                           r"\text{Std: } &45.65")
        
        ga_gd_stats = MathTex(r"\text{Mean: } &0.11\\"
                                r"\text{Median: } &0.05\\"
                                r"\text{Std: } &0.18")
        
        stats = VGroup(gd_stats, ga_gd_stats)
        stats.arrange(buff=0.5)
        stats.shift(DOWN*3)
        gd_stats.scale(0.5)
        ga_gd_stats.scale(0.5)
        
        self.play(FadeIn(genetic_alg_comparison_plot), Write(stats))
        

        self.next_slide()
        self.play(FadeOut(genetic_alg_comparison_plot), FadeOut(title), FadeOut(stats))
        

        second_approach = Text("Second Approach", font_size=50)
        self.play(Write(second_approach))
        title = Text("Carleman Matrices", font_size=50)
        self.play(ReplacementTransform(second_approach, title))
        self.play(title.animate.shift(UP*3))

        def_carleman = MathTex(
            r"M[f]_{jk} = \frac{1}{k!}[\frac{d^k}{dx^k}(f(x))^j]_{x=0}")
        def_carleman.next_to(title, DOWN, buff=0.5)
        self.play(Write(def_carleman))

        self.next_slide()

        example = MathTex(r"f(x) = 3 - x + 2x^2")
        example_carleman = MathTex(r"M[f] = \begin{bmatrix}"
        r"1 & 0 & 0 & 0 & 0 \\"
        r"3 & -1 & 2 & 0 & 0 \\"
        r"9 & -6 & 13 & -4 & 4 \\"
        r"27 & -27 & 63 & -37 & 42 \\"
        r"81 & -108 & 270 & -228 & 289 \\"
        r"\end{bmatrix}")
        
        example_group = VGroup(example, example_carleman)
        example.next_to(def_carleman, DOWN, buff=0.5)
        example_carleman.next_to(example, DOWN, buff=0.5)
        self.play(Write(example_group))

        self.next_slide()
        self.play(example[0][5].animate.set_color(YELLOW), example[0][9].animate.set_color(YELLOW),
                  example_carleman[0][16].animate.set_color(YELLOW), example_carleman[0][17:19].animate.set_color(YELLOW), example_carleman[0][19].animate.set_color(YELLOW))

        self.next_slide()
        self.play(FadeOut(def_carleman), FadeOut(example_group), FadeOut(title))
        mg = MathTex(r"M[g]")
        mh = MathTex(r"M[h]")
        property_carleman_lhs = VGroup(mg, mh)
        property_carleman_lhs.arrange(buff=0.05)
        eq = MathTex(r"=")
        property_carleman_rhs = MathTex(r"M[g \circ h]")
        property_carleman = VGroup(property_carleman_lhs, eq, property_carleman_rhs)
        property_carleman.arrange(buff=0.2)
        #property_carleman.next_to(def_carleman, DOWN, buff=0.5)
        self.play(Write(property_carleman))
        
        self.next_slide()
        want = MathTex(r"\text{We want to find } g \text{ and } h \text{ such that } M[g \circ h] = M[Q]")
        want.shift(DOWN*2)
        self.play(Write(want))
        mq = MathTex(r"M[Q]")
        mq.next_to(eq, RIGHT, buff=0.2)
        self.play(ReplacementTransform(property_carleman_rhs, mq))
        
        self.next_slide()
        self.play(FadeOut(want))
        coefs_def_1 = MathTex(r"\text{We define the coefficients } g_i, h_i, \text{ and } q_i \text{ as }")
        coefs_def_2 = MathTex(r"\text{the coefficients of polynomials } g, h \text{ and } Q \text{ respectively}")
        coefs_def = VGroup(coefs_def_1, coefs_def_2)
        coefs_def.arrange(DOWN)
        coefs_def.shift(DOWN*2)
        
        self.play(Write(coefs_def))
        
        mg_expanded = MathTex(r"\begin{bmatrix}"
        r"1 & 0 & 0 & 0 & 0 & 0 \\"
        r"g_0 & g_1 & \ldots & g_n & 0 & \ldots \\"
        r"\vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\"
        r"\end{bmatrix}")
        
        mq_expanded = MathTex(r"\begin{bmatrix}"
        r"1 & 0 & 0 & 0 & 0 & 0 \\"
        r"q_0 & q_1 & \ldots & q_{mn} & 0 & \ldots \\"
        r"\vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\"
        r"\end{bmatrix}")
        
        mg_expanded.scale(0.8).next_to(mh, LEFT, buff=0.2)
        mq_expanded.scale(0.8).next_to(eq, buff=0.2)
        
        self.play(ReplacementTransform(mg, mg_expanded), ReplacementTransform(mq, mq_expanded))

        self.next_slide()
        new_equation = MathTex(r"M[h]^T \begin{bmatrix} g_0 \\ g_1 \\ \vdots \\ g_n \end{bmatrix} = \begin{bmatrix} q_0 \\ q_1 \\ \vdots \\ q_{mn} \end{bmatrix}")
        self.play(ReplacementTransform(property_carleman, new_equation), FadeOut(coefs_def))
        
        self.next_slide()
        least_squares = MathTex(r"\text{If we know } M[h] \text{, we can solve for } g \text{ using least squares}")
        least_squares.next_to(new_equation, DOWN, buff=0.5)
        self.play(Write(least_squares))

        self.next_slide()
        self.play(FadeOut(least_squares), FadeOut(new_equation))
        
        assume_g = MathTex(r"\text{If we instead assume that } g \text{ is known}")
        assume_g.shift(UP*2)
        self.play(Write(assume_g))
        
        mg = MathTex(r"M[g]")
        mh = MathTex(r"M[h]")
        property_carleman_lhs = VGroup(mg, mh)
        property_carleman_lhs.arrange(buff=0.05)
        eq = MathTex(r"=")
        property_carleman_rhs = MathTex(r"M[Q]")
        property_carleman = VGroup(property_carleman_lhs, eq, property_carleman_rhs)
        property_carleman.arrange(buff=0.2)
        self.play(Write(property_carleman))
        
        self.next_slide()
        mg_inv = MathTex(r"M[g]^{-1}")
        mg_inv.next_to(eq, RIGHT, buff=0.2)
        
        self.play(ReplacementTransform(mg, mg_inv), property_carleman_rhs.animate.next_to(mg_inv, RIGHT, buff=0.05))

        self.next_slide()
        what_if = MathTex(r"\text{What if } M[g] \text{ is not invertible? In general, it isn't.}")
        moore_penrose = MathTex(r"\text{We can use the Moore-Penrose pseudoinverse } M[g]^+")
        what_if.next_to(property_carleman, DOWN, buff=0.5)
        moore_penrose.next_to(what_if, DOWN, buff=0.5)
        self.play(Write(what_if))
        self.play(Write(moore_penrose))

        mg_pseudoinverse = MathTex(r"M[g]^+")
        kinda_eq = MathTex(r"\approx")
        kinda_eq.next_to(mh, RIGHT, buff=0.2)
        mg_pseudoinverse.next_to(eq, RIGHT, buff=0.2)
        self.play(ReplacementTransform(mg_inv, mg_pseudoinverse), ReplacementTransform(eq, kinda_eq), property_carleman_rhs.animate.next_to(mg_pseudoinverse, RIGHT, buff=0.05))

        self.next_slide()
        self.play(FadeOut(what_if), FadeOut(moore_penrose))
        
        mht = MathTex(r"M[h]^T")
        mqt = MathTex(r"M[Q]^T")
        mgplust = MathTex(r"(M[g]^+)^T")
        mht.next_to(kinda_eq, LEFT, buff=0.2)
        mqt.next_to(kinda_eq, buff=0.2)
        mgplust.next_to(mqt, buff=0.05)
        
        self.play(ReplacementTransform(mh, mht), ReplacementTransform(mg_pseudoinverse, mgplust), ReplacementTransform(property_carleman_rhs, mqt))
        
        mh_row2 = MathTex(r"\begin{bmatrix} h_0 \\ h_1 \\ \vdots \\ h_n \end{bmatrix}")
        mg_pseudoinverse_row2 = MathTex(r"\begin{bmatrix} (M[g]^+)_{2, 1} \\ (M[g]^+)_{2, 2} \\ \vdots \\ (M[g]^+)_{2, m} \end{bmatrix}")

        mh_row2.next_to(kinda_eq, LEFT, buff=0.2)
        mg_pseudoinverse_row2.next_to(property_carleman_rhs, RIGHT, buff=0.2)

        self.play(ReplacementTransform(mht, mh_row2), ReplacementTransform(mgplust, mg_pseudoinverse_row2))
        
        self.next_slide()
        self.play(FadeOut(mh_row2), FadeOut(mg_pseudoinverse_row2), FadeOut(mqt), FadeOut(kinda_eq), FadeOut(assume_g))
        
        # Explaining full carleman algo
        title = Text("Full algorithm")
        title.shift(UP*3)
        self.play(Write(title))
        
        guess_h = MathTex(r"\text{1. Guess some } h")
        use_least_squares = MathTex(r"\text{2. Use least squares to find the best } g \text{ for that } h")
        use_pseudo_inverse = MathTex(r"\text{3. Use the Moore-Penrose pseudoinverse to find the best } h \text{ for that } g")
        iterate = MathTex(r"\text{4. Iterate until convergence}")
        
        steps = VGroup(guess_h, use_least_squares, use_pseudo_inverse, iterate)
        steps.arrange(DOWN, aligned_edge=LEFT).next_to(title, DOWN, buff=0.5).scale(0.8)
        
        self.next_slide()
        self.play(Write(guess_h))
        
        self.next_slide()
        self.play(Write(use_least_squares))
        
        self.next_slide()
        self.play(Write(use_pseudo_inverse))
        
        self.next_slide()
        self.play(Write(iterate))
        
        self.next_slide()
        self.play(FadeOut(steps), FadeOut(title))

        h = P.Polynomial([-0.48438477, 0.17471423, 0.30691025, -0.47821644])
        g = P.Polynomial([-0.30220919, 0.45056399, -0.12502072, -0.34156207])
        target = P.Polynomial([-0.74312614, 0.58070565, -1.38637741, -0.24022258,
                               -0.12865073, -0.27551531, -0.01900289, 0.079588,
                               0.03172626, 0.00359399])
        
        def polynomial_to_latex(coeffs):
            terms = []
            for i, coeff in enumerate(coeffs):
                if coeff != 0:
                    term = f"{coeff:.4f}"
                    if i == 1:
                        term += "x"
                    elif i > 1:
                        term += f"x^{i}"
                    terms.append(term)
            return " + ".join(terms).replace("+ -", "- ")
        
        h_mob = MathTex(f"h(x) = {polynomial_to_latex(h.coef)}")
        g_mob = MathTex(f"g(x) = {polynomial_to_latex(g.coef)}")
        goh_mob = MathTex(f"g(h(x)) = {polynomial_to_latex(compose_layers([h, g]).coef)}", tex_to_color_map={"g": YELLOW, "h": YELLOW})
        target_mob = MathTex(f"Q(x) = {polynomial_to_latex(target.coef)}", tex_to_color_map={"Q": BLUE})
        
        polys = VGroup(h_mob, g_mob, goh_mob, target_mob)
        polys.arrange(DOWN, aligned_edge=LEFT).scale(0.5).shift(UP*3)
        
        l_inf_error_text = MathTex(r"\|Q(x) - g(h(x))\|_{L^\infty} = ", tex_to_color_map={"Q": BLUE, "g": YELLOW, "h": YELLOW})
        l_inf_error = MathTex(f"{l_inf_norm(target, compose_layers([h, g])):.4f}")
        l_inf_error_group = VGroup(l_inf_error_text, l_inf_error)
        l_inf_error_group.arrange(buff=0.2)
        l_inf_error_group.shift(DOWN*3)
        
        iter_text = MathTex(r"\text{Iteration}")
        iter_nbr = MathTex("0")
        iter_grp = VGroup(iter_text, iter_nbr)
        iter_nbr.next_to(iter_text, RIGHT, buff=0.1)
        iter_grp.next_to(l_inf_error_group, UP, buff=0.25)
        
        # Create axes
        axes = Axes(
            x_range=[-0.1, 1.1],
            y_range=[-2, 0.1],
            axis_config={"color": BLUE}
        )
        # Create the target polynomial plot
        target_plot = axes.plot(lambda x: target(x), color=BLUE, use_smoothing=False, x_range=[0, 1, 0.001])

        # Create the composed polynomial plot
        composed_plot = axes.plot(lambda x: h(g(x)), color=YELLOW, use_smoothing=False, x_range=[0, 1, 0.001])

        plots = VGroup(axes, target_plot, composed_plot)
        plots.scale(0.8)

        # Add the plots to the scene
        self.play(Create(axes), Create(target_plot), Create(composed_plot), Write(polys), Write(l_inf_error_group), Write(iter_grp))

        self.next_slide()

        for i in range(1, 6):
            h, g = carleman_solver(h, g, target, iteration=i)
            composed = compose_layers([h, g])
            new_composed_plot = axes.plot(lambda x: composed(x), color=YELLOW, use_smoothing=False, x_range=[0, 1, 0.001])

            new_l_inf_error = MathTex(f"{l_inf_norm(target, composed):.4f}")
            new_l_inf_error.next_to(l_inf_error_text, RIGHT, buff=0.2)

            new_iter_nbr = MathTex(f"{i}")
            new_iter_nbr.next_to(iter_text, RIGHT, buff=0.2)

            new_h_mob = MathTex(f"h(x) = {polynomial_to_latex(h.coef)}")
            new_g_mob = MathTex(f"g(x) = {polynomial_to_latex(g.coef)}")
            new_goh_mob = MathTex(f"g(h(x)) = {polynomial_to_latex(composed.coef)}", tex_to_color_map={"g": YELLOW, "h": YELLOW})
            new_target_mob = MathTex(f"Q(x) = {polynomial_to_latex(target.coef)}", tex_to_color_map={"Q": BLUE})
            new_polys = VGroup(new_h_mob, new_g_mob, new_goh_mob, new_target_mob)
            new_polys.arrange(DOWN, aligned_edge=LEFT).scale(0.5).shift(UP*3)

            self.play(ReplacementTransform(iter_nbr, new_iter_nbr), ReplacementTransform(composed_plot, new_composed_plot),
                      ReplacementTransform(l_inf_error, new_l_inf_error), ReplacementTransform(h_mob, new_h_mob),
                      ReplacementTransform(g_mob, new_g_mob), ReplacementTransform(goh_mob, new_goh_mob),
                      ReplacementTransform(target_mob, new_target_mob))
            l_inf_error = new_l_inf_error
            composed_plot = new_composed_plot
            iter_nbr = new_iter_nbr
            h_mob = new_h_mob
            g_mob = new_g_mob
            goh_mob = new_goh_mob
            target_mob = new_target_mob
        
        self.next_slide()
        self.play(FadeOut(polys), FadeOut(composed_plot), FadeOut(target_plot), FadeOut(axes),
                  FadeOut(l_inf_error_group), FadeOut(iter_grp))
        
        title = Text("Observations")
        title.shift(UP*3)
        self.play(Write(title))
        
        speed = Text("Carleman runs in about 0.1s (unoptimized), against 10s for gradient descent", font_size=20)
        reliability = Text("But it only works maybe 1 in 5 times", font_size=20)
        norm = Text("And it optimizes a stupid norm", font_size=20)
        
        speed.next_to(title, DOWN, buff=0.5)
        reliability.next_to(speed, DOWN, buff=0.5)
        norm.next_to(reliability, DOWN, buff=0.5)
        
        self.next_slide()
        self.play(Write(speed))
        
        self.next_slide()
        self.play(Write(reliability))
        
        self.next_slide()
        self.play(Write(norm))
        
        self.next_slide()
        self.play(FadeOut(title), FadeOut(speed), FadeOut(reliability), FadeOut(norm))
        questions = Text("Questions?", font_size=50)
        self.play(Write(questions))
