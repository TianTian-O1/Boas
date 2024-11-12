	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 12, 0
	.section	__TEXT,__literal8,8byte_literals
	.p2align	3, 0x0                          ## -- Begin function main
LCPI0_0:
	.quad	0x4033000000000000              ## double 19
LCPI0_1:
	.quad	0x4036000000000000              ## double 22
LCPI0_2:
	.quad	0x4045800000000000              ## double 43
LCPI0_3:
	.quad	0x4049000000000000              ## double 50
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_main
	.p2align	4
_main:                                  ## @main
	.cfi_startproc
## %bb.0:                               ## %entry
	subq	$56, %rsp
	.cfi_def_cfa_offset 64
	movabsq	$4611686019492741120, %rax      ## imm = 0x400000003F800000
	movq	%rax, 40(%rsp)
	movabsq	$4647714816524288000, %rax      ## imm = 0x4080000040400000
	movq	%rax, 48(%rsp)
	movabsq	$4665729215040061440, %rax      ## imm = 0x40C0000040A00000
	movq	%rax, 24(%rsp)
	movabsq	$4683743613553737728, %rax      ## imm = 0x4100000040E00000
	movq	%rax, 32(%rsp)
	movabsq	$4733283209466871808, %rax      ## imm = 0x41B0000041980000
	movq	%rax, 8(%rsp)
	movabsq	$4776067405936590848, %rax      ## imm = 0x42480000422C0000
	movq	%rax, 16(%rsp)
	leaq	L_format(%rip), %rdi
	movsd	LCPI0_0(%rip), %xmm0            ## xmm0 = [1.9E+1,0.0E+0]
	movsd	LCPI0_1(%rip), %xmm1            ## xmm1 = [2.2E+1,0.0E+0]
	movsd	LCPI0_2(%rip), %xmm2            ## xmm2 = [4.3E+1,0.0E+0]
	movsd	LCPI0_3(%rip), %xmm3            ## xmm3 = [5.0E+1,0.0E+0]
	movb	$4, %al
	callq	_printf
	xorl	%eax, %eax
	addq	$56, %rsp
	retq
	.cfi_endproc
                                        ## -- End function
	.section	__TEXT,__cstring,cstring_literals
L_format:                               ## @format
	.asciz	"Matrix:\n[%.0f, %.0f]\n[%.0f, %.0f]\n"

.subsections_via_symbols
