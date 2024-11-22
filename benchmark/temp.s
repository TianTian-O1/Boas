	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 12, 0
	.section	__TEXT,__literal8,8byte_literals
	.p2align	3, 0x0                          ## -- Begin function main
LCPI0_0:
	.quad	0x3ff0000000000000              ## double 1
LCPI0_1:
	.quad	0x4000000000000000              ## double 2
LCPI0_2:
	.quad	0x4008000000000000              ## double 3
LCPI0_3:
	.quad	0x4010000000000000              ## double 4
LCPI0_4:
	.quad	0x4014000000000000              ## double 5
LCPI0_5:
	.quad	0x4018000000000000              ## double 6
LCPI0_6:
	.quad	0x401c000000000000              ## double 7
LCPI0_7:
	.quad	0x4020000000000000              ## double 8
LCPI0_8:
	.quad	0x4022000000000000              ## double 9
LCPI0_9:
	.quad	0x4024000000000000              ## double 10
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_main
	.p2align	4
_main:                                  ## @main
	.cfi_startproc
## %bb.0:
	pushq	%r14
	.cfi_def_cfa_offset 16
	pushq	%rbx
	.cfi_def_cfa_offset 24
	pushq	%rax
	.cfi_def_cfa_offset 32
	.cfi_offset %rbx, -24
	.cfi_offset %r14, -16
	vmovsd	LCPI0_0(%rip), %xmm0            ## xmm0 = [1.0E+0,0.0E+0]
	callq	_printFloat
	xorl	%ebx, %ebx
	jmp	LBB0_1
	.p2align	4
LBB0_5:                                 ##   in Loop: Header=BB0_1 Depth=1
	incq	%rbx
LBB0_1:                                 ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_4 Depth 2
	cmpq	$63, %rbx
	jg	LBB0_6
## %bb.2:                               ## %.preheader29
                                        ##   in Loop: Header=BB0_1 Depth=1
	xorl	%r14d, %r14d
	cmpq	$63, %r14
	jg	LBB0_5
	.p2align	4
LBB0_4:                                 ##   Parent Loop BB0_1 Depth=1
                                        ## =>  This Inner Loop Header: Depth=2
	callq	_generate_random
	incq	%r14
	cmpq	$63, %r14
	jle	LBB0_4
	jmp	LBB0_5
LBB0_6:                                 ## %.preheader28
	xorl	%ebx, %ebx
	jmp	LBB0_7
	.p2align	4
LBB0_11:                                ##   in Loop: Header=BB0_7 Depth=1
	incq	%rbx
LBB0_7:                                 ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_10 Depth 2
	cmpq	$63, %rbx
	jg	LBB0_12
## %bb.8:                               ## %.preheader27
                                        ##   in Loop: Header=BB0_7 Depth=1
	xorl	%r14d, %r14d
	cmpq	$63, %r14
	jg	LBB0_11
	.p2align	4
LBB0_10:                                ##   Parent Loop BB0_7 Depth=1
                                        ## =>  This Inner Loop Header: Depth=2
	callq	_generate_random
	incq	%r14
	cmpq	$63, %r14
	jle	LBB0_10
	jmp	LBB0_11
LBB0_12:                                ## %.preheader26
	xorl	%eax, %eax
	jmp	LBB0_13
	.p2align	4
LBB0_20:                                ##   in Loop: Header=BB0_13 Depth=1
	incq	%rax
LBB0_13:                                ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_15 Depth 2
                                        ##       Child Loop BB0_18 Depth 3
	cmpq	$63, %rax
	jg	LBB0_21
## %bb.14:                              ## %.preheader25
                                        ##   in Loop: Header=BB0_13 Depth=1
	xorl	%ecx, %ecx
	jmp	LBB0_15
	.p2align	4
LBB0_19:                                ##   in Loop: Header=BB0_15 Depth=2
	incq	%rcx
LBB0_15:                                ##   Parent Loop BB0_13 Depth=1
                                        ## =>  This Loop Header: Depth=2
                                        ##       Child Loop BB0_18 Depth 3
	cmpq	$63, %rcx
	jg	LBB0_20
## %bb.16:                              ## %.preheader24
                                        ##   in Loop: Header=BB0_15 Depth=2
	xorl	%edx, %edx
	cmpq	$63, %rdx
	jg	LBB0_19
	.p2align	4
LBB0_18:                                ##   Parent Loop BB0_13 Depth=1
                                        ##     Parent Loop BB0_15 Depth=2
                                        ## =>    This Inner Loop Header: Depth=3
	incq	%rdx
	cmpq	$63, %rdx
	jle	LBB0_18
	jmp	LBB0_19
LBB0_21:
	vmovsd	LCPI0_1(%rip), %xmm0            ## xmm0 = [2.0E+0,0.0E+0]
	callq	_printFloat
	vmovsd	LCPI0_2(%rip), %xmm0            ## xmm0 = [3.0E+0,0.0E+0]
	callq	_printFloat
	xorl	%ebx, %ebx
	jmp	LBB0_22
	.p2align	4
LBB0_26:                                ##   in Loop: Header=BB0_22 Depth=1
	incq	%rbx
LBB0_22:                                ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_25 Depth 2
	cmpq	$127, %rbx
	jg	LBB0_27
## %bb.23:                              ## %.preheader23
                                        ##   in Loop: Header=BB0_22 Depth=1
	xorl	%r14d, %r14d
	cmpq	$127, %r14
	jg	LBB0_26
	.p2align	4
LBB0_25:                                ##   Parent Loop BB0_22 Depth=1
                                        ## =>  This Inner Loop Header: Depth=2
	callq	_generate_random
	incq	%r14
	cmpq	$127, %r14
	jle	LBB0_25
	jmp	LBB0_26
LBB0_27:                                ## %.preheader22
	xorl	%ebx, %ebx
	jmp	LBB0_28
	.p2align	4
LBB0_32:                                ##   in Loop: Header=BB0_28 Depth=1
	incq	%rbx
LBB0_28:                                ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_31 Depth 2
	cmpq	$127, %rbx
	jg	LBB0_33
## %bb.29:                              ## %.preheader21
                                        ##   in Loop: Header=BB0_28 Depth=1
	xorl	%r14d, %r14d
	cmpq	$127, %r14
	jg	LBB0_32
	.p2align	4
LBB0_31:                                ##   Parent Loop BB0_28 Depth=1
                                        ## =>  This Inner Loop Header: Depth=2
	callq	_generate_random
	incq	%r14
	cmpq	$127, %r14
	jle	LBB0_31
	jmp	LBB0_32
LBB0_33:                                ## %.preheader20
	xorl	%eax, %eax
	jmp	LBB0_34
	.p2align	4
LBB0_41:                                ##   in Loop: Header=BB0_34 Depth=1
	incq	%rax
LBB0_34:                                ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_36 Depth 2
                                        ##       Child Loop BB0_39 Depth 3
	cmpq	$127, %rax
	jg	LBB0_42
## %bb.35:                              ## %.preheader19
                                        ##   in Loop: Header=BB0_34 Depth=1
	xorl	%ecx, %ecx
	jmp	LBB0_36
	.p2align	4
LBB0_40:                                ##   in Loop: Header=BB0_36 Depth=2
	incq	%rcx
LBB0_36:                                ##   Parent Loop BB0_34 Depth=1
                                        ## =>  This Loop Header: Depth=2
                                        ##       Child Loop BB0_39 Depth 3
	cmpq	$127, %rcx
	jg	LBB0_41
## %bb.37:                              ## %.preheader18
                                        ##   in Loop: Header=BB0_36 Depth=2
	xorl	%edx, %edx
	cmpq	$127, %rdx
	jg	LBB0_40
	.p2align	4
LBB0_39:                                ##   Parent Loop BB0_34 Depth=1
                                        ##     Parent Loop BB0_36 Depth=2
                                        ## =>    This Inner Loop Header: Depth=3
	incq	%rdx
	cmpq	$127, %rdx
	jle	LBB0_39
	jmp	LBB0_40
LBB0_42:
	vmovsd	LCPI0_3(%rip), %xmm0            ## xmm0 = [4.0E+0,0.0E+0]
	callq	_printFloat
	vmovsd	LCPI0_4(%rip), %xmm0            ## xmm0 = [5.0E+0,0.0E+0]
	callq	_printFloat
	xorl	%ebx, %ebx
	jmp	LBB0_43
	.p2align	4
LBB0_47:                                ##   in Loop: Header=BB0_43 Depth=1
	incq	%rbx
LBB0_43:                                ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_46 Depth 2
	cmpq	$255, %rbx
	jg	LBB0_48
## %bb.44:                              ## %.preheader17
                                        ##   in Loop: Header=BB0_43 Depth=1
	xorl	%r14d, %r14d
	cmpq	$255, %r14
	jg	LBB0_47
	.p2align	4
LBB0_46:                                ##   Parent Loop BB0_43 Depth=1
                                        ## =>  This Inner Loop Header: Depth=2
	callq	_generate_random
	incq	%r14
	cmpq	$255, %r14
	jle	LBB0_46
	jmp	LBB0_47
LBB0_48:                                ## %.preheader16
	xorl	%ebx, %ebx
	jmp	LBB0_49
	.p2align	4
LBB0_53:                                ##   in Loop: Header=BB0_49 Depth=1
	incq	%rbx
LBB0_49:                                ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_52 Depth 2
	cmpq	$255, %rbx
	jg	LBB0_54
## %bb.50:                              ## %.preheader15
                                        ##   in Loop: Header=BB0_49 Depth=1
	xorl	%r14d, %r14d
	cmpq	$255, %r14
	jg	LBB0_53
	.p2align	4
LBB0_52:                                ##   Parent Loop BB0_49 Depth=1
                                        ## =>  This Inner Loop Header: Depth=2
	callq	_generate_random
	incq	%r14
	cmpq	$255, %r14
	jle	LBB0_52
	jmp	LBB0_53
LBB0_54:                                ## %.preheader14
	xorl	%eax, %eax
	jmp	LBB0_55
	.p2align	4
LBB0_62:                                ##   in Loop: Header=BB0_55 Depth=1
	incq	%rax
LBB0_55:                                ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_57 Depth 2
                                        ##       Child Loop BB0_60 Depth 3
	cmpq	$255, %rax
	jg	LBB0_63
## %bb.56:                              ## %.preheader13
                                        ##   in Loop: Header=BB0_55 Depth=1
	xorl	%ecx, %ecx
	jmp	LBB0_57
	.p2align	4
LBB0_61:                                ##   in Loop: Header=BB0_57 Depth=2
	incq	%rcx
LBB0_57:                                ##   Parent Loop BB0_55 Depth=1
                                        ## =>  This Loop Header: Depth=2
                                        ##       Child Loop BB0_60 Depth 3
	cmpq	$255, %rcx
	jg	LBB0_62
## %bb.58:                              ## %.preheader12
                                        ##   in Loop: Header=BB0_57 Depth=2
	xorl	%edx, %edx
	cmpq	$255, %rdx
	jg	LBB0_61
	.p2align	4
LBB0_60:                                ##   Parent Loop BB0_55 Depth=1
                                        ##     Parent Loop BB0_57 Depth=2
                                        ## =>    This Inner Loop Header: Depth=3
	incq	%rdx
	cmpq	$255, %rdx
	jle	LBB0_60
	jmp	LBB0_61
LBB0_63:
	vmovsd	LCPI0_5(%rip), %xmm0            ## xmm0 = [6.0E+0,0.0E+0]
	callq	_printFloat
	vmovsd	LCPI0_6(%rip), %xmm0            ## xmm0 = [7.0E+0,0.0E+0]
	callq	_printFloat
	xorl	%ebx, %ebx
	jmp	LBB0_64
	.p2align	4
LBB0_68:                                ##   in Loop: Header=BB0_64 Depth=1
	incq	%rbx
LBB0_64:                                ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_67 Depth 2
	cmpq	$511, %rbx                      ## imm = 0x1FF
	jg	LBB0_69
## %bb.65:                              ## %.preheader11
                                        ##   in Loop: Header=BB0_64 Depth=1
	xorl	%r14d, %r14d
	cmpq	$511, %r14                      ## imm = 0x1FF
	jg	LBB0_68
	.p2align	4
LBB0_67:                                ##   Parent Loop BB0_64 Depth=1
                                        ## =>  This Inner Loop Header: Depth=2
	callq	_generate_random
	incq	%r14
	cmpq	$511, %r14                      ## imm = 0x1FF
	jle	LBB0_67
	jmp	LBB0_68
LBB0_69:                                ## %.preheader10
	xorl	%ebx, %ebx
	jmp	LBB0_70
	.p2align	4
LBB0_74:                                ##   in Loop: Header=BB0_70 Depth=1
	incq	%rbx
LBB0_70:                                ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_73 Depth 2
	cmpq	$511, %rbx                      ## imm = 0x1FF
	jg	LBB0_75
## %bb.71:                              ## %.preheader9
                                        ##   in Loop: Header=BB0_70 Depth=1
	xorl	%r14d, %r14d
	cmpq	$511, %r14                      ## imm = 0x1FF
	jg	LBB0_74
	.p2align	4
LBB0_73:                                ##   Parent Loop BB0_70 Depth=1
                                        ## =>  This Inner Loop Header: Depth=2
	callq	_generate_random
	incq	%r14
	cmpq	$511, %r14                      ## imm = 0x1FF
	jle	LBB0_73
	jmp	LBB0_74
LBB0_75:                                ## %.preheader8
	xorl	%eax, %eax
	jmp	LBB0_76
	.p2align	4
LBB0_83:                                ##   in Loop: Header=BB0_76 Depth=1
	incq	%rax
LBB0_76:                                ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_78 Depth 2
                                        ##       Child Loop BB0_81 Depth 3
	cmpq	$511, %rax                      ## imm = 0x1FF
	jg	LBB0_84
## %bb.77:                              ## %.preheader7
                                        ##   in Loop: Header=BB0_76 Depth=1
	xorl	%ecx, %ecx
	jmp	LBB0_78
	.p2align	4
LBB0_82:                                ##   in Loop: Header=BB0_78 Depth=2
	incq	%rcx
LBB0_78:                                ##   Parent Loop BB0_76 Depth=1
                                        ## =>  This Loop Header: Depth=2
                                        ##       Child Loop BB0_81 Depth 3
	cmpq	$511, %rcx                      ## imm = 0x1FF
	jg	LBB0_83
## %bb.79:                              ## %.preheader6
                                        ##   in Loop: Header=BB0_78 Depth=2
	xorl	%edx, %edx
	cmpq	$511, %rdx                      ## imm = 0x1FF
	jg	LBB0_82
	.p2align	4
LBB0_81:                                ##   Parent Loop BB0_76 Depth=1
                                        ##     Parent Loop BB0_78 Depth=2
                                        ## =>    This Inner Loop Header: Depth=3
	incq	%rdx
	cmpq	$511, %rdx                      ## imm = 0x1FF
	jle	LBB0_81
	jmp	LBB0_82
LBB0_84:
	vmovsd	LCPI0_7(%rip), %xmm0            ## xmm0 = [8.0E+0,0.0E+0]
	callq	_printFloat
	vmovsd	LCPI0_8(%rip), %xmm0            ## xmm0 = [9.0E+0,0.0E+0]
	callq	_printFloat
	xorl	%ebx, %ebx
	jmp	LBB0_85
	.p2align	4
LBB0_89:                                ##   in Loop: Header=BB0_85 Depth=1
	incq	%rbx
LBB0_85:                                ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_88 Depth 2
	cmpq	$1023, %rbx                     ## imm = 0x3FF
	jg	LBB0_90
## %bb.86:                              ## %.preheader5
                                        ##   in Loop: Header=BB0_85 Depth=1
	xorl	%r14d, %r14d
	cmpq	$1023, %r14                     ## imm = 0x3FF
	jg	LBB0_89
	.p2align	4
LBB0_88:                                ##   Parent Loop BB0_85 Depth=1
                                        ## =>  This Inner Loop Header: Depth=2
	callq	_generate_random
	incq	%r14
	cmpq	$1023, %r14                     ## imm = 0x3FF
	jle	LBB0_88
	jmp	LBB0_89
LBB0_90:                                ## %.preheader4
	xorl	%ebx, %ebx
	jmp	LBB0_91
	.p2align	4
LBB0_95:                                ##   in Loop: Header=BB0_91 Depth=1
	incq	%rbx
LBB0_91:                                ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_94 Depth 2
	cmpq	$1023, %rbx                     ## imm = 0x3FF
	jg	LBB0_96
## %bb.92:                              ## %.preheader3
                                        ##   in Loop: Header=BB0_91 Depth=1
	xorl	%r14d, %r14d
	cmpq	$1023, %r14                     ## imm = 0x3FF
	jg	LBB0_95
	.p2align	4
LBB0_94:                                ##   Parent Loop BB0_91 Depth=1
                                        ## =>  This Inner Loop Header: Depth=2
	callq	_generate_random
	incq	%r14
	cmpq	$1023, %r14                     ## imm = 0x3FF
	jle	LBB0_94
	jmp	LBB0_95
LBB0_96:                                ## %.preheader2
	xorl	%eax, %eax
	jmp	LBB0_97
	.p2align	4
LBB0_104:                               ##   in Loop: Header=BB0_97 Depth=1
	incq	%rax
LBB0_97:                                ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_99 Depth 2
                                        ##       Child Loop BB0_102 Depth 3
	cmpq	$1023, %rax                     ## imm = 0x3FF
	jg	LBB0_105
## %bb.98:                              ## %.preheader1
                                        ##   in Loop: Header=BB0_97 Depth=1
	xorl	%ecx, %ecx
	jmp	LBB0_99
	.p2align	4
LBB0_103:                               ##   in Loop: Header=BB0_99 Depth=2
	incq	%rcx
LBB0_99:                                ##   Parent Loop BB0_97 Depth=1
                                        ## =>  This Loop Header: Depth=2
                                        ##       Child Loop BB0_102 Depth 3
	cmpq	$1023, %rcx                     ## imm = 0x3FF
	jg	LBB0_104
## %bb.100:                             ## %.preheader
                                        ##   in Loop: Header=BB0_99 Depth=2
	xorl	%edx, %edx
	cmpq	$1023, %rdx                     ## imm = 0x3FF
	jg	LBB0_103
	.p2align	4
LBB0_102:                               ##   Parent Loop BB0_97 Depth=1
                                        ##     Parent Loop BB0_99 Depth=2
                                        ## =>    This Inner Loop Header: Depth=3
	incq	%rdx
	cmpq	$1023, %rdx                     ## imm = 0x3FF
	jle	LBB0_102
	jmp	LBB0_103
LBB0_105:
	vmovsd	LCPI0_9(%rip), %xmm0            ## xmm0 = [1.0E+1,0.0E+0]
	callq	_printFloat
	xorl	%eax, %eax
	addq	$8, %rsp
	popq	%rbx
	popq	%r14
	retq
	.cfi_endproc
                                        ## -- End function
.subsections_via_symbols
